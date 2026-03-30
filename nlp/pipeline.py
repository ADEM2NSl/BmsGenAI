"""
nlp/pipeline.py
Bilingual NLP Pipeline for BMS requirements (DE + EN)
spaCy de_core_news_sm + en_core_web_sm
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional
from loguru import logger

import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────

STOP_EN = set(stopwords.words("english")) | {
    "unit", "dim", "1x1", "must", "bmw", "bmu", "hl", "rc",
    "equal", "equals", "output", "value", "values", "input", "inputs",
}
STOP_DE = set(stopwords.words("german")) | {
    "muss", "wird", "soll", "kann", "bzw", "sowie", "gem", "wert",
    "ausgeben", "eingang", "ausgang", "bmw", "bmu", "hl", "rc",
    "müssen", "werden", "sollen",
}

CRITICAL_PATTERNS_EN = {
    "range_check":  r"\b(range|in range|boundary|limit|min|max|threshold)\b",
    "fault":        r"\b(fault|error|failure|invalid|implausible|defect|diagnostic)\b",
    "safety":       r"\b(safety|critical|emergency|protection|shutdown|overvoltage)\b",
    "timing":       r"\b(timeout|delay|debouncing|debounce|cycle|period)\b",
    "plausibility": r"\b(plausib|qual_int|qualifier|valid|not ok)\b",
}
CRITICAL_PATTERNS_DE = {
    "bereich":        r"\b(bereich|grenze|grenzwert|minimum|maximum|schwellwert)\b",
    "fehler":         r"\b(fehler|defekt|ungültig|nicht plausibel|ausfall|störung)\b",
    "sicherheit":     r"\b(sicherheit|kritisch|notfall|schutz|abschaltung)\b",
    "plausibilitaet": r"\b(plausib|qualifier|gültig|n\.i\.o|nicht i\.o)\b",
    "bedingung":      r"\b(wenn|falls|sofern|außer|sonst|sobald)\b",
}
QUALITY_PATTERNS_EN = {
    "vague_terms":  r"\b(appropriate|adequate|sufficient|reasonable|normal|sometimes)\b",
    "no_threshold": r"(?<![0-9])\b(high|low|large|small|fast|slow)\b",
    "missing_unit": r"\b(temperature|voltage|current|time)\b(?!.*°|.*[Vv]olt|.*[Aa]mp)",
}
QUALITY_PATTERNS_DE = {
    "vage_begriffe":    r"\b(geeignet|ausreichend|angemessen|normal|manchmal)\b",
    "kein_schwellwert": r"(?<![0-9])\b(hoch|niedrig|groß|klein|schnell|langsam)\b",
    "fehlende_einheit": r"\b(temperatur|spannung|strom|zeit)\b(?!.*°|.*[Vv]olt|.*[Aa]mp)",
}

TOPIC_LABELS_EN = [
    "Temperature Monitoring",
    "Fault & Plausibility",
    "Signal Output",
    "OBD Data",
    "Range & Conditions",
]
TOPIC_LABELS_DE = [
    "Temperaturüberwachung",
    "Fehler & Plausibilität",
    "Signalausgabe",
    "OBD-Daten",
    "Bereich & Bedingungen",
]

# ── NLP Pipeline ──────────────────────────────────────────────────────────────

class BMSNLPPipeline:
    def __init__(self):
        logger.info("Loading spaCy models...")
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_de = spacy.load("de_core_news_sm")
        logger.info("spaCy models loaded ✅")

    # ── Preprocessing ─────────────────────────────────────────────────

    @staticmethod
    def clean_bms(text: str) -> str:
        text = re.sub(r"BMW_\w+", "", text)
        text = re.sub(r"QUAL_[A-Z_]+", "", text)
        text = re.sub(r"-?\d+\.?\d*\s*°?[cCkK%]?", "", text)
        text = re.sub(r"\(Unit:[^)]*\)", "", text)
        text = re.sub(r"\(Dim:[^)]*\)", "", text)
        return text

    def preprocess_en(self, text: str) -> list[str]:
        doc = self.nlp_en(self.clean_bms(text).lower())
        return [
            t.lemma_ for t in doc
            if t.is_alpha and len(t.text) > 2
            and t.text not in STOP_EN and not t.is_stop
        ]

    def preprocess_de(self, text: str) -> list[str]:
        doc = self.nlp_de(self.clean_bms(text).lower())
        return [
            t.lemma_ for t in doc
            if t.is_alpha and len(t.text) > 2
            and t.text not in STOP_DE and not t.is_stop
        ]

    # ── Entity Extraction ─────────────────────────────────────────────

    @staticmethod
    def extract_bms_entities(text: str) -> dict:
        return {
            "signals":    list(set(re.findall(r"BMW_[A-Za-z0-9_]+", text))),
            "thresholds": re.findall(r"(-?\d+\.?\d*)\s*°C", text),
            "qualifiers": list(set(re.findall(r"QUAL_[A-Z_]+", text))),
            "conditions": re.findall(r"\b(AND|OR|IF|WHEN|UNLESS)\b", text),
            "error_vals": re.findall(r"BMW_LIM_[A-Za-z0-9_]+", text),
        }

    def extract_spacy_ner(self, text: str, lang: str = "en") -> list[tuple]:
        nlp = self.nlp_en if lang == "en" else self.nlp_de
        doc = nlp(text[:5000])
        return [(e.text, e.label_) for e in doc.ents]

    # ── Criticality Scoring ───────────────────────────────────────────

    @staticmethod
    def score_criticality(text_en: str, text_de: str) -> dict:
        scores = {}
        for name, pat in CRITICAL_PATTERNS_EN.items():
            scores[f"flag_en_{name}"] = int(bool(
                re.search(pat, text_en, re.IGNORECASE)
            ))
        for name, pat in CRITICAL_PATTERNS_DE.items():
            scores[f"flag_de_{name}"] = int(bool(
                re.search(pat, text_de, re.IGNORECASE)
            ))
        total = sum(scores.values())
        scores["criticality_score"] = total
        scores["is_critical"] = total >= 3
        return scores

    # ── Quality Analysis ──────────────────────────────────────────────

    @staticmethod
    def analyze_quality(text_en: str, text_de: str) -> dict:
        issues = {}
        for name, pat in QUALITY_PATTERNS_EN.items():
            issues[f"issue_en_{name}"] = int(bool(
                re.search(pat, text_en, re.IGNORECASE)
            ))
        for name, pat in QUALITY_PATTERNS_DE.items():
            issues[f"issue_de_{name}"] = int(bool(
                re.search(pat, text_de, re.IGNORECASE)
            ))
        issues["quality_issues"] = sum(issues.values())
        return issues

    # ── ECU Integration Classification ───────────────────────────────

    @staticmethod
    def classify_ecu_level(text_en: str, text_de: str) -> str:
        text = (text_en + " " + text_de).lower()
        if re.search(r"\b(OBD|ITID|PID|DTC|UDS|diagnostic)\b", text, re.IGNORECASE):
            return "OBD / Diagnostics"
        elif re.search(r"\b(CAN|LIN|PDU|bus|frame|message)\b", text, re.IGNORECASE):
            return "CAN Bus / Network"
        elif re.search(r"\b(cycle|zykl|task|ms|period|scheduling)\b", text, re.IGNORECASE):
            return "Timing / Scheduling"
        elif re.search(r"\b(qualifier|QUAL_INT|plausib|sensor)\b", text, re.IGNORECASE):
            return "Signal Plausibility"
        else:
            return "Software Unit"

    # ── Main Run ─────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full NLP pipeline on a requirements DataFrame.
        Expects columns: text_en, text_de
        """
        logger.info(f"Running NLP pipeline on {len(df)} requirements...")
        df = df.copy()

        # Text lengths
        df["len_words_en"] = df["text_en"].apply(lambda x: len(x.split()))
        df["len_words_de"] = df["text_de"].apply(lambda x: len(x.split()))
        df["cond_en"] = df["text_en"].apply(
            lambda x: len(re.findall(r"\b(AND|OR|IF|WHEN|if|when|and|or)\b", x))
        )
        df["cond_de"] = df["text_de"].apply(
            lambda x: len(re.findall(r"\b(UND|ODER|WENN|FALLS|und|oder|wenn|falls)\b", x))
        )

        # Preprocessing
        logger.info("Preprocessing text (DE + EN)...")
        df["tokens_en"] = df["text_en"].apply(self.preprocess_en)
        df["tokens_de"] = df["text_de"].apply(self.preprocess_de)
        df["clean_en"] = df["tokens_en"].apply(lambda t: " ".join(t))
        df["clean_de"] = df["tokens_de"].apply(lambda t: " ".join(t))

        # BMS entities
        logger.info("Extracting BMS entities...")
        df["bms_entities"] = df["text_en"].apply(self.extract_bms_entities)
        df["n_signals"]    = df["bms_entities"].apply(lambda e: len(e["signals"]))
        df["n_thresholds"] = df["bms_entities"].apply(lambda e: len(e["thresholds"]))
        df["n_qualifiers"] = df["bms_entities"].apply(lambda e: len(e["qualifiers"]))

        # Criticality
        logger.info("Scoring criticality...")
        crit = df.apply(
            lambda r: self.score_criticality(r["text_en"], r["text_de"]), axis=1
        )
        df = pd.concat([df, pd.DataFrame(list(crit))], axis=1)

        # Quality
        logger.info("Analyzing quality...")
        qual = df.apply(
            lambda r: self.analyze_quality(r["text_en"], r["text_de"]), axis=1
        )
        df = pd.concat([df, pd.DataFrame(list(qual))], axis=1)

        # ECU level
        df["ecu_level"] = df.apply(
            lambda r: self.classify_ecu_level(r["text_en"], r["text_de"]), axis=1
        )

        # LDA topic modeling
        logger.info("Running LDA topic modeling...")
        df = self._run_lda(df)

        # Clustering
        logger.info("Running K-Means clustering...")
        df = self._run_clustering(df)

        logger.info("NLP pipeline complete ✅")
        return df

    def _run_lda(self, df: pd.DataFrame, n_topics: int = 5) -> pd.DataFrame:
        for lang, col, labels in [
            ("en", "clean_en", TOPIC_LABELS_EN),
            ("de", "clean_de", TOPIC_LABELS_DE),
        ]:
            vec = TfidfVectorizer(max_features=100, min_df=1)
            X = vec.fit_transform(df[col])
            lda = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, max_iter=20
            )
            doc_topics = lda.fit_transform(X)
            df[f"topic_{lang}"] = doc_topics.argmax(axis=1)
            df[f"topic_label_{lang}"] = df[f"topic_{lang}"].map(
                lambda i: labels[i]
            )
        return df

    def _run_clustering(self, df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        CLUSTER_LABELS = [
            "Thermal / Thermisch",
            "OBD Diagnostics / Diagnose",
            "Fault Handling / Fehler",
            "Plausibility / Plausibilität",
        ]
        combined = df["clean_en"] + " " + df["clean_de"]
        vec = TfidfVectorizer(max_features=200, min_df=1)
        X = normalize(vec.fit_transform(combined).toarray())
        df["cluster"] = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        ).fit_predict(X)
        df["cluster_label"] = df["cluster"].map(lambda c: CLUSTER_LABELS[c])
        return df
