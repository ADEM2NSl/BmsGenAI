"""
backend/ingestion.py
Loads and validates BMS requirement files (Excel / PDF)
"""

import re
import pandas as pd
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, field_validator


# ── Pydantic validation schema ─────────────────────────────────────────────

class RequirementRow(BaseModel):
    item_id: int
    section: str
    description_de: str
    description_en: str
    status: str
    typ: str

    @field_validator("status")
    @classmethod
    def check_status(cls, v):
        allowed = {"Freigegeben", "Entwurf", "Ungültig", "In Bearbeitung"}
        if v not in allowed:
            raise ValueError(f"Unknown status: {v}")
        return v


# ── Loader ─────────────────────────────────────────────────────────────────

class BMSRequirementLoader:

    REQUIRED_COLUMNS = [
        "Item ID", "Gliederungsnummer", "Beschreibung",
        "Englisch", "Typ", "Status",
    ]

    def load_excel(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        logger.info(f"Loading Excel: {path.name}")

        # Try skiprows=3 (standard HVSCC format)
        df = pd.read_excel(path, skiprows=3)

        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter actual requirements
        df = df[df["Typ"] == "Anforderung"].copy().reset_index(drop=True)

        # Normalize text columns
        df["text_de"] = df["Beschreibung"].fillna("")
        df["text_en"] = df["Englisch"].fillna("")

        logger.info(f"Loaded {len(df)} requirements from {path.name}")
        return df

    def load_pdf(self, path: str | Path) -> str:
        """Extract text from a BMS spec PDF."""
        path = Path(path)
        logger.info(f"Loading PDF: {path.name}")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            logger.info(f"Extracted {len(text)} chars from {path.name}")
            return text
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return ""

    def validate(self, df: pd.DataFrame) -> dict:
        """Run basic validation checks on loaded requirements."""
        issues = []

        # Missing text
        empty_de = (df["text_de"] == "").sum()
        empty_en = (df["text_en"] == "").sum()
        if empty_de > 0:
            issues.append(f"{empty_de} requirements missing German text")
        if empty_en > 0:
            issues.append(f"{empty_en} requirements missing English text")

        # Duplicate sections
        dups = df["Gliederungsnummer"].duplicated().sum()
        if dups > 0:
            issues.append(f"{dups} duplicate section numbers")

        # Entwurf count
        drafts = (df["Status"] == "Entwurf").sum()

        return {
            "total": len(df),
            "approved": (df["Status"] == "Freigegeben").sum(),
            "draft": drafts,
            "issues": issues,
            "valid": len(issues) == 0,
        }
