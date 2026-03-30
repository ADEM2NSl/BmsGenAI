"""
tests/test_pipeline.py
Pytest test suite for BMS GenAI Assistant
Run with: pytest tests/ -v --tb=short
"""

import sys
import json
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingestion import BMSRequirementLoader
from nlp.pipeline import BMSNLPPipeline
from nlp.test_generator import BMSTestCaseGenerator

SAMPLE_FILE = Path("data/uploads/HVSCC-38653_AM_Final.xlsx")


@pytest.fixture(scope="module")
def sample_df():
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    loader = BMSRequirementLoader()
    return loader.load_excel(SAMPLE_FILE)


@pytest.fixture(scope="module")
def enriched_df(sample_df):
    nlp = BMSNLPPipeline()
    return nlp.run(sample_df)


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestIngestion:
    def test_load_excel(self, sample_df):
        assert len(sample_df) > 0
        assert "text_de" in sample_df.columns
        assert "text_en" in sample_df.columns

    def test_required_columns(self, sample_df):
        required = ["Item ID", "Gliederungsnummer", "Status", "Typ"]
        for col in required:
            assert col in sample_df.columns, f"Missing: {col}"

    def test_only_anforderung(self, sample_df):
        assert (sample_df["Typ"] == "Anforderung").all()

    def test_validation(self, sample_df):
        loader = BMSRequirementLoader()
        result = loader.validate(sample_df)
        assert "total" in result
        assert result["total"] == len(sample_df)
        assert result["approved"] > 0


# ── NLP Pipeline tests ────────────────────────────────────────────────────────

class TestNLPPipeline:
    def test_tokens_generated(self, enriched_df):
        assert "tokens_en" in enriched_df.columns
        assert "tokens_de" in enriched_df.columns
        assert enriched_df["tokens_en"].apply(len).sum() > 0

    def test_criticality_columns(self, enriched_df):
        assert "criticality_score" in enriched_df.columns
        assert "is_critical" in enriched_df.columns
        assert enriched_df["criticality_score"].dtype in ["int64", "float64"]

    def test_some_critical(self, enriched_df):
        assert enriched_df["is_critical"].sum() > 0, "Expected at least 1 critical requirement"

    def test_topics_assigned(self, enriched_df):
        assert "topic_label_en" in enriched_df.columns
        assert "topic_label_de" in enriched_df.columns
        assert enriched_df["topic_label_en"].nunique() > 1

    def test_clusters_assigned(self, enriched_df):
        assert "cluster_label" in enriched_df.columns
        assert enriched_df["cluster_label"].nunique() > 1

    def test_bms_entities(self, enriched_df):
        assert "bms_entities" in enriched_df.columns
        # At least some requirements should have signals
        has_signals = enriched_df["bms_entities"].apply(
            lambda e: len(e.get("signals", [])) > 0 if isinstance(e, dict) else False
        )
        assert has_signals.sum() > 0

    def test_ecu_level(self, enriched_df):
        assert "ecu_level" in enriched_df.columns
        valid_levels = {
            "OBD / Diagnostics", "CAN Bus / Network",
            "Timing / Scheduling", "Signal Plausibility", "Software Unit"
        }
        assert set(enriched_df["ecu_level"].unique()).issubset(valid_levels)

    def test_quality_columns(self, enriched_df):
        issue_cols = [c for c in enriched_df.columns if c.startswith("issue_")]
        assert len(issue_cols) > 0


# ── Test Case Generator tests ─────────────────────────────────────────────────

class TestTestCaseGenerator:
    def test_unit_tcs_generated(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, _ = gen.run(enriched_df)
        assert len(df_unit) > 0

    def test_ecu_tcs_generated(self, enriched_df):
        gen = BMSTestCaseGenerator()
        _, df_ecu = gen.run(enriched_df)
        assert len(df_ecu) > 0

    def test_unit_tc_columns(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, _ = gen.run(enriched_df)
        required = ["TC_ID", "Type_EN", "Type_DE", "Objective_EN", "Ziel_DE",
                    "Expected_Result_EN", "Priority"]
        for col in required:
            assert col in df_unit.columns, f"Missing column: {col}"

    def test_ecu_tc_columns(self, enriched_df):
        gen = BMSTestCaseGenerator()
        _, df_ecu = gen.run(enriched_df)
        required = ["TC_ID", "Integration_Type_EN", "Test_Environment",
                    "Objective_EN", "Expected_Result_EN", "Measurement_Tool"]
        for col in required:
            assert col in df_ecu.columns, f"Missing column: {col}"

    def test_tc_types_present(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, _ = gen.run(enriched_df)
        types = set(df_unit["Type_EN"].unique())
        assert "Nominal" in types
        assert "Fault" in types or "Boundary" in types

    def test_ecu_types_present(self, enriched_df):
        gen = BMSTestCaseGenerator()
        _, df_ecu = gen.run(enriched_df)
        types = set(df_ecu["Integration_Type_EN"].unique())
        assert "End-to-End Signal Flow" in types

    def test_no_null_tc_ids(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, df_ecu = gen.run(enriched_df)
        assert df_unit["TC_ID"].notna().all()
        assert df_ecu["TC_ID"].notna().all()

    def test_unique_tc_ids(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, df_ecu = gen.run(enriched_df)
        assert df_unit["TC_ID"].nunique() == len(df_unit), "Duplicate unit TC IDs!"
        assert df_ecu["TC_ID"].nunique() == len(df_ecu), "Duplicate ECU TC IDs!"

    def test_bilingual_fields(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, _ = gen.run(enriched_df)
        assert df_unit["Objective_EN"].notna().all()
        assert df_unit["Ziel_DE"].notna().all()
        assert df_unit["Expected_Result_EN"].notna().all()
        assert df_unit["Erwartetes_Ergebnis_DE"].notna().all()

    def test_priority_values(self, enriched_df):
        gen = BMSTestCaseGenerator()
        df_unit, df_ecu = gen.run(enriched_df)
        valid_prios = {"Critical", "High", "Medium", "Low"}
        assert set(df_unit["Priority"].unique()).issubset(valid_prios)
        assert set(df_ecu["Priority"].unique()).issubset(valid_prios)


# ── Entity extraction tests ───────────────────────────────────────────────────

class TestEntityExtraction:
    def test_signal_extraction(self):
        text = "BMW_t_HvsCellMinObd must equal BMW_t_HvsCellCoreAct_rc value"
        entities = BMSNLPPipeline.extract_bms_entities(text)
        assert "BMW_t_HvsCellMinObd" in entities["signals"]

    def test_threshold_extraction(self):
        text = "Value must be between -40°C and 215°C"
        entities = BMSNLPPipeline.extract_bms_entities(text)
        assert "-40" in entities["thresholds"]
        assert "215" in entities["thresholds"]

    def test_qualifier_extraction(self):
        text = "Qualifier QUAL_INT_OK or QUAL_INT_DEBOUNCING"
        entities = BMSNLPPipeline.extract_bms_entities(text)
        assert "QUAL_INT_OK" in entities["qualifiers"]
        assert "QUAL_INT_DEBOUNCING" in entities["qualifiers"]

    def test_criticality_scorer(self):
        text_en = "fault detection for safety-critical boundary limit check"
        text_de = "Fehler und Plausibilitätsprüfung am Grenzwert"
        scores = BMSNLPPipeline.score_criticality(text_en, text_de)
        assert scores["criticality_score"] >= 3
        assert scores["is_critical"] is True
