"""
backend/database.py
SQLite + SQLAlchemy database layer for BMS GenAI Assistant
"""

import json
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class RequirementRun(Base):
    __tablename__ = "requirement_runs"
    id          = Column(String, primary_key=True)
    filename    = Column(String)
    created_at  = Column(DateTime, default=datetime.utcnow)
    req_count   = Column(Integer)
    critical_count = Column(Integer)
    data_json   = Column(Text)  # serialized DataFrame


class TestCaseRun(Base):
    __tablename__ = "testcase_runs"
    id          = Column(String, primary_key=True)
    req_run_id  = Column(String)
    created_at  = Column(DateTime, default=datetime.utcnow)
    unit_count  = Column(Integer)
    ecu_count   = Column(Integer)
    llm_count   = Column(Integer)
    unit_json   = Column(Text)
    ecu_json    = Column(Text)
    llm_json    = Column(Text)
    approved_ids = Column(Text, default="[]")


class BMSDatabase:
    def __init__(self, db_path: str = "./data/bms.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {db_path}")

    # ── Requirements ────────────────────────────────────────────────

    def save_requirements(self, df: pd.DataFrame, filename: str) -> str:
        run_id = str(uuid.uuid4())[:8]
        df_save = df.copy()

        # Convert every column to a JSON-safe type
        for col in df_save.columns:
            # Convert bool columns explicitly first (numpy bool_ -> Python bool)
            if df_save[col].dtype == bool or str(df_save[col].dtype) == "bool":
                df_save[col] = df_save[col].astype(bool)
            # Convert numpy int/float
            elif df_save[col].dtype in ["int64", "int32", "float64", "float32"]:
                df_save[col] = df_save[col].astype(float)
            # Convert object columns (lists, dicts) to string
            elif df_save[col].dtype == object:
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list)) else str(x)
                )

        with Session(self.engine) as session:
            run = RequirementRun(
                id=run_id,
                filename=filename,
                req_count=int(len(df)),
                critical_count=int(df["is_critical"].sum()) if "is_critical" in df.columns else 0,
                data_json=df_save.to_json(orient="records", force_ascii=False),
            )
            session.add(run)
            session.commit()
        logger.info(f"Saved requirement run: {run_id}")
        return run_id

    def get_requirements(self, run_id: str) -> Optional[dict]:
        with Session(self.engine) as session:
            run = session.get(RequirementRun, run_id)
            if not run:
                return None
            return {
                "run_id": run.id,
                "filename": run.filename,
                "created_at": run.created_at.isoformat(),
                "req_count": run.req_count,
                "critical_count": run.critical_count,
                "requirements": json.loads(run.data_json),
            }

    def get_requirements_df(self, run_id: str) -> Optional[pd.DataFrame]:
        data = self.get_requirements(run_id)
        if not data:
            return None
        df = pd.DataFrame(data["requirements"])
        # Restore list columns from string
        for col in ["bms_entities", "tokens_en", "tokens_de"]:
            if col in df.columns:
                try:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except Exception:
                    pass
        return df

    def get_stats(self, run_id: str) -> dict:
        data = self.get_requirements(run_id)
        if not data:
            return {}
        df = pd.DataFrame(data["requirements"])
        stats = {
            "run_id": run_id,
            "total": int(len(df)),
            "approved": int((df["Status"] == "Freigegeben").sum()) if "Status" in df.columns else 0,
            "critical": int(df["is_critical"].sum()) if "is_critical" in df.columns else 0,
        }
        if "topic_label_en" in df.columns:
            stats["topics_en"] = {k: int(v) for k, v in df["topic_label_en"].value_counts().items()}
        if "cluster_label" in df.columns:
            stats["clusters"] = {k: int(v) for k, v in df["cluster_label"].value_counts().items()}
        if "ecu_level" in df.columns:
            stats["ecu_levels"] = {k: int(v) for k, v in df["ecu_level"].value_counts().items()}
        if "topic_label_de" in df.columns:
            stats["topics_de"] = {k: int(v) for k, v in df["topic_label_de"].value_counts().items()}
        return stats
        return stats

    def list_runs(self) -> list[dict]:
        with Session(self.engine) as session:
            runs = session.query(RequirementRun).order_by(
                RequirementRun.created_at.desc()
            ).all()
            return [
                {
                    "run_id": r.id,
                    "filename": r.filename,
                    "created_at": r.created_at.isoformat(),
                    "req_count": r.req_count,
                    "critical_count": r.critical_count,
                }
                for r in runs
            ]

    # ── Test Cases ──────────────────────────────────────────────────

    def save_test_cases(
        self,
        req_run_id: str,
        df_unit: pd.DataFrame,
        df_ecu: pd.DataFrame,
        llm_tcs: list = None,
    ) -> str:
        tc_id = str(uuid.uuid4())[:8]
        llm_tcs = llm_tcs or []

        def df_to_json(df):
            return df.to_json(orient="records", force_ascii=False) if len(df) else "[]"

        with Session(self.engine) as session:
            run = TestCaseRun(
                id=tc_id,
                req_run_id=req_run_id,
                unit_count=len(df_unit),
                ecu_count=len(df_ecu),
                llm_count=len(llm_tcs),
                unit_json=df_to_json(df_unit),
                ecu_json=df_to_json(df_ecu),
                llm_json=json.dumps(llm_tcs, ensure_ascii=False),
                approved_ids="[]",
            )
            session.add(run)
            session.commit()
        logger.info(f"Saved TC run: {tc_id}")
        return tc_id

    def get_test_cases(self, tc_run_id: str) -> Optional[dict]:
        with Session(self.engine) as session:
            run = session.get(TestCaseRun, tc_run_id)
            if not run:
                return None
            return {
                "tc_run_id": run.id,
                "req_run_id": run.req_run_id,
                "created_at": run.created_at.isoformat(),
                "unit_count": run.unit_count,
                "ecu_count": run.ecu_count,
                "llm_count": run.llm_count,
                "unit_tcs": json.loads(run.unit_json),
                "ecu_tcs": json.loads(run.ecu_json),
                "llm_tcs": json.loads(run.llm_json),
                "approved_ids": json.loads(run.approved_ids),
            }

    def approve_test_cases(self, tc_run_id: str, tc_ids: list[str], approved_by: str):
        with Session(self.engine) as session:
            run = session.get(TestCaseRun, tc_run_id)
            if run:
                current = json.loads(run.approved_ids)
                updated = list(set(current + tc_ids))
                run.approved_ids = json.dumps(updated)
                session.commit()
        logger.info(f"Approved {len(tc_ids)} TCs in run {tc_run_id} by {approved_by}")

    # ── Exports ─────────────────────────────────────────────────────

    def export_excel(self, tc_run_id: str, output_dir: Path) -> Optional[str]:
        data = self.get_test_cases(tc_run_id)
        if not data:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"bms_testcases_{ts}.xlsx"

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            pd.DataFrame(data["unit_tcs"]).to_excel(writer, sheet_name="Unit Tests", index=False)
            pd.DataFrame(data["ecu_tcs"]).to_excel(writer, sheet_name="ECU Integration", index=False)
            if data["llm_tcs"]:
                pd.DataFrame(data["llm_tcs"]).to_excel(writer, sheet_name="LLM Generated", index=False)

        logger.info(f"Exported Excel: {path}")
        return str(path)

    def export_canoe_xml(self, tc_run_id: str, output_dir: Path) -> Optional[str]:
        """Export ECU integration test cases using ECU.TEST Object API / XML format."""
        data = self.get_test_cases(tc_run_id)
        if not data:
            return None

        from ecu.ecutest_integration import ECUTestExporter
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        exporter = ECUTestExporter()
        df_unit = pd.DataFrame(data["unit_tcs"])
        df_ecu  = pd.DataFrame(data["ecu_tcs"])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ecu_out_dir = output_dir / f"ecutest_{ts}"

        prj_path = exporter.export_project_xml(
            df_unit, df_ecu, ecu_out_dir,
            project_name=f"BMS_TestSuite_{ts}"
        )
        logger.info(f"Exported ECU.TEST project: {prj_path}")
        return prj_path
