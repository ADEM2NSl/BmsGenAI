"""
cicd/run_pipeline.py
Run the full BMS pipeline locally (no GitLab needed).
Usage: python cicd/run_pipeline.py data/uploads/HVSCC-38653_AM_Final.xlsx
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pipeline(xlsx_path: str, use_llm: bool = False):
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("BMS GenAI Assistant — Local Pipeline")
    logger.info(f"File:    {xlsx_path}")
    logger.info(f"LLM:     {'enabled' if use_llm else 'disabled'}")
    logger.info(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    base_name = Path(xlsx_path).stem

    # ── Stage 1: Load & Validate ─────────────────────────────────────
    logger.info("\n[STAGE 1] Load & Validate Requirements")
    from backend.ingestion import BMSRequirementLoader
    loader = BMSRequirementLoader()
    df = loader.load_excel(xlsx_path)
    validation = loader.validate(df)
    logger.info(f"  Total:    {validation['total']}")
    logger.info(f"  Approved: {validation['approved']}")
    logger.info(f"  Draft:    {validation['draft']}")
    if validation["issues"]:
        for issue in validation["issues"]:
            logger.warning(f"  ⚠️  {issue}")
    else:
        logger.success("  ✅ Validation passed")

    # ── Stage 2: NLP Pipeline ────────────────────────────────────────
    logger.info("\n[STAGE 2] NLP Pipeline (DE + EN)")
    from nlp.pipeline import BMSNLPPipeline
    nlp = BMSNLPPipeline()
    df_enriched = nlp.run(df)

    nlp_out = out_dir / f"nlp_{base_name}.json"
    df_enriched.to_json(nlp_out, orient="records", force_ascii=False)
    logger.success(f"  ✅ NLP complete → {nlp_out}")
    logger.info(f"  Critical: {df_enriched['is_critical'].sum()}/{len(df_enriched)}")

    # ── Stage 3: LLM Generation ──────────────────────────────────────
    llm_tcs = []
    if use_llm:
        logger.info("\n[STAGE 3] LLM Test Case Generation (Ollama/Mistral)")
        from llm.generator import BMSLLMGenerator
        llm = BMSLLMGenerator()
        if llm.is_available():
            llm.index_requirements(df_enriched.to_dict("records"))
            for _, row in df_enriched.iterrows():
                tcs = llm.generate_test_cases(row.to_dict())
                llm_tcs.extend(tcs)
            logger.success(f"  ✅ LLM generated {len(llm_tcs)} additional TCs")
        else:
            logger.warning("  ⚠️  Ollama not available — skipping LLM stage")
            logger.warning("  Run: ollama serve && ollama pull mistral")
    else:
        logger.info("\n[STAGE 3] LLM Generation — SKIPPED (use --llm to enable)")

    # ── Stage 4: Test Case Generation ────────────────────────────────
    logger.info("\n[STAGE 4] Test Case Generation (Unit + ECU Integration)")
    from nlp.test_generator import BMSTestCaseGenerator
    import pandas as pd
    tc_gen = BMSTestCaseGenerator()
    df_unit, df_ecu = tc_gen.run(df_enriched)

    unit_out = out_dir / f"unit_tcs_{base_name}.xlsx"
    ecu_out  = out_dir / f"ecu_tcs_{base_name}.xlsx"
    df_unit.to_excel(unit_out, index=False)
    df_ecu.to_excel(ecu_out, index=False)
    logger.success(f"  ✅ Unit TCs ({len(df_unit)}) → {unit_out}")
    logger.success(f"  ✅ ECU TCs  ({len(df_ecu)}) → {ecu_out}")

    # ── Stage 5: Combined Excel + ECU.TEST Export ────────────────────
    logger.info("\n[STAGE 5] Export")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_out = out_dir / f"bms_testcases_{base_name}_{ts}.xlsx"

    with pd.ExcelWriter(combined_out, engine="xlsxwriter") as writer:
        df_unit.to_excel(writer, sheet_name="Unit Tests",       index=False)
        df_ecu.to_excel(writer,  sheet_name="ECU Integration",  index=False)
        if llm_tcs:
            pd.DataFrame(llm_tcs).to_excel(writer, sheet_name="LLM Generated", index=False)

        summary = pd.DataFrame([
            {"Metric": "Total Requirements",      "Value": len(df_enriched)},
            {"Metric": "Critical Requirements",   "Value": int(df_enriched["is_critical"].sum())},
            {"Metric": "Unit Test Cases",         "Value": len(df_unit)},
            {"Metric": "ECU Integration TCs",     "Value": len(df_ecu)},
            {"Metric": "LLM Generated TCs",       "Value": len(llm_tcs)},
            {"Metric": "Total Test Cases",        "Value": len(df_unit) + len(df_ecu) + len(llm_tcs)},
            {"Metric": "Generated At",            "Value": ts},
        ])
        summary.to_excel(writer, sheet_name="Summary", index=False)

    logger.success(f"  ✅ Combined Excel → {combined_out}")

    # ECU.TEST export (packages + project file)
    logger.info("  Generating ECU.TEST packages...")
    from ecu.ecutest_integration import ECUTestIntegration
    ecutest = ECUTestIntegration()
    ecu_result = ecutest.generate_and_export(
        df_unit, df_ecu,
        output_dir=out_dir / f"ecutest_{base_name}",
        project_name=f"BMS_{base_name}"
    )
    logger.success(f"  ✅ ECU.TEST packages ({ecu_result['package_count']}) → {ecu_result['output_dir']}")

    # ── Stage 6: Coverage Report ─────────────────────────────────────
    logger.info("\n[STAGE 6] Coverage Report")
    from cicd.generate_report import generate_report
    generate_report()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).seconds
    logger.info("\n" + "=" * 60)
    logger.success("PIPELINE COMPLETE ✅")
    logger.info(f"  Time elapsed:   {elapsed}s")
    logger.info(f"  Requirements:   {len(df_enriched)}")
    logger.info(f"  Critical:       {df_enriched['is_critical'].sum()}")
    logger.info(f"  Unit TCs:       {len(df_unit)}")
    logger.info(f"  ECU TCs:        {len(df_ecu)}")
    logger.info(f"  LLM TCs:        {len(llm_tcs)}")
    logger.info(f"  Total TCs:      {len(df_unit) + len(df_ecu) + len(llm_tcs)}")
    logger.info(f"\n  Outputs in:  ./outputs/")
    logger.info("=" * 60)

    return {
        "requirements": len(df_enriched),
        "critical": int(df_enriched["is_critical"].sum()),
        "unit_tcs": len(df_unit),
        "ecu_tcs": len(df_ecu),
        "llm_tcs": len(llm_tcs),
        "output": str(combined_out),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMS GenAI Pipeline")
    parser.add_argument("file", help="Path to requirements Excel file")
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation")
    args = parser.parse_args()
    run_pipeline(args.file, use_llm=args.llm)
