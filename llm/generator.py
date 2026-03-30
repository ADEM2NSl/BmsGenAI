"""
llm/generator.py
LLM integration via Ollama — supports cloud models:
  - deepseek-v3.1:671b-cloud  (best for technical/structured output)
  - qwen3-coder:480b-cloud    (best for code/JSON generation)
  - qwen3-vl:235b-cloud       (multimodal)
  - gpt-oss:120b-cloud        (GPT-class)
  - gpt-oss:20b-cloud         (fast GPT-class)

Plus RAG using ChromaDB + sentence-transformers.
"""

import json
import re
import os
from typing import Optional
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not found — LLM generation disabled")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not found — RAG disabled")


# ── Supported Models ──────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    # Cloud models (from your Ollama instance)
    "deepseek-v3.1:671b-cloud": {
        "description": "DeepSeek V3.1 671B — Best for technical structured output",
        "temperature": 0.1,   # Very low — extremely deterministic
        "num_predict": 3000,  # Large context window
        "think": False,       # No chain-of-thought needed for structured JSON
        "recommended_for": "test case generation",
    },
    "qwen3-coder:480b-cloud": {
        "description": "Qwen3 Coder 480B — Best for JSON/code structured output",
        "temperature": 0.1,
        "num_predict": 3000,
        "think": False,
        "recommended_for": "test case generation",
    },
    "qwen3-vl:235b-cloud": {
        "description": "Qwen3 VL 235B — Multimodal, good for technical analysis",
        "temperature": 0.15,
        "num_predict": 2500,
        "think": False,
        "recommended_for": "analysis + generation",
    },
    "gpt-oss:120b-cloud": {
        "description": "GPT-class 120B — High quality, balanced",
        "temperature": 0.2,
        "num_predict": 2000,
        "think": False,
        "recommended_for": "test case generation",
    },
    "gpt-oss:20b-cloud": {
        "description": "GPT-class 20B — Fastest cloud model",
        "temperature": 0.2,
        "num_predict": 2000,
        "think": False,
        "recommended_for": "quick generation",
    },
    # Local models (fallback)
    "mistral": {
        "description": "Mistral 7B — Local, no internet needed",
        "temperature": 0.2,
        "num_predict": 2000,
        "think": False,
        "recommended_for": "offline use",
    },
    "llama3": {
        "description": "LLaMA 3 8B — Local fallback",
        "temperature": 0.2,
        "num_predict": 2000,
        "think": False,
        "recommended_for": "offline use",
    },
}

DEFAULT_MODEL = "deepseek-v3.1:671b-cloud"

# ── System Prompt (model-agnostic) ────────────────────────────────────────────

BMS_SYSTEM_PROMPT = """You are an expert automotive validation engineer specializing in
Battery Management Systems (BMS) for electric vehicles.
You generate precise, structured, bilingual test cases for BMS ECU software validation.

STRICT RULES:
- Respond with a valid JSON array ONLY — no markdown fences, no explanation, no preamble
- Every test case MUST have both German (DE) and English (EN) fields
- Use exact signal names from the requirement (BMW_* format)
- Use exact qualifier names (QUAL_INT_OK, QUAL_INT_DEBOUNCING)
- Use exact temperature values in °C
- Follow ISO 26262 functional safety principles
- Priority must be exactly one of: Critical, High, Medium, Low

OUTPUT FORMAT — JSON array:
[
  {
    "TC_ID": "TC_<section>_<number>",
    "Type_EN": "Nominal|Boundary|Fault|Out-of-Range",
    "Type_DE": "Normalfall|Grenzwert|Fehlerfall|Bereichsüberschreitung",
    "Objective_EN": "<clear test objective in English>",
    "Ziel_DE": "<klares Testziel auf Deutsch>",
    "Preconditions_EN": "<setup conditions in English>",
    "Vorbedingungen_DE": "<Vorbedingungen auf Deutsch>",
    "Expected_Result_EN": "<expected output in English>",
    "Erwartetes_Ergebnis_DE": "<erwartetes Ergebnis auf Deutsch>",
    "Priority": "Critical|High|Medium|Low",
    "Rationale": "<why this TC is needed>"
  }
]"""

# DeepSeek / Qwen think-mode system prompt (no thinking tags in output)
BMS_SYSTEM_PROMPT_DEEPSEEK = BMS_SYSTEM_PROMPT + """

IMPORTANT: Do not output <think>...</think> blocks. Output ONLY the JSON array."""

FEW_SHOT_EXAMPLE = """
=== EXAMPLE ===

REQUIREMENT:
The BMU HL must output the minimum OBD cell core temperature
BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreAct_rc.BMW_t_HvsCellCoreMinAct
when qualifier is OK (QUAL_INT_OK or QUAL_INT_DEBOUNCING)
and value is in range: -40°C to 215°C.
Otherwise set to BMW_LIM_MAXERRTEMP_SC.

EXPECTED OUTPUT:
[
  {
    "TC_ID": "TC_17_8_001",
    "Type_EN": "Nominal",
    "Type_DE": "Normalfall",
    "Objective_EN": "Verify BMW_t_HvsCellMinObd equals input when qualifier OK and value in valid range",
    "Ziel_DE": "BMW_t_HvsCellMinObd entspricht Eingangswert wenn Qualifier i.O. und Wert im gültigen Bereich",
    "Preconditions_EN": "BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK; input in range [-40°C, 215°C]",
    "Vorbedingungen_DE": "BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK; Eingangswert im Bereich [-40°C, 215°C]",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
    "Priority": "High",
    "Rationale": "Core nominal behavior — must pass for basic requirement compliance"
  },
  {
    "TC_ID": "TC_17_8_010",
    "Type_EN": "Boundary",
    "Type_DE": "Grenzwert",
    "Objective_EN": "Verify output at lower boundary -40°C",
    "Ziel_DE": "Ausgabe am unteren Grenzwert -40°C prüfen",
    "Preconditions_EN": "Input = -40°C; BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK",
    "Vorbedingungen_DE": "Eingangswert = -40°C; BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = -40°C; no error value set",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = -40°C; kein Fehlerwert gesetzt",
    "Priority": "High",
    "Rationale": "Boundary value at minimum valid temperature"
  },
  {
    "TC_ID": "TC_17_8_099",
    "Type_EN": "Fault",
    "Type_DE": "Fehlerfall",
    "Objective_EN": "Verify error value set when qualifier is invalid",
    "Ziel_DE": "Fehlerwert-Setzung bei ungültigem Qualifier prüfen",
    "Preconditions_EN": "BMW_t_HvsCellCoreAct_qi != QUAL_INT_OK AND != QUAL_INT_DEBOUNCING",
    "Vorbedingungen_DE": "BMW_t_HvsCellCoreAct_qi != QUAL_INT_OK UND != QUAL_INT_DEBOUNCING",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = BMW_LIM_MAXERRTEMP_SC; output not updated",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = BMW_LIM_MAXERRTEMP_SC; Ausgangswert nicht aktualisiert",
    "Priority": "Critical",
    "Rationale": "Safety-critical: invalid qualifier must trigger safe error value"
  }
]

=== END EXAMPLE ===
"""


class BMSLLMGenerator:
    """
    BMS Test Case Generator using Ollama cloud/local models.
    Recommended: deepseek-v3.1:671b-cloud or qwen3-coder:480b-cloud
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_host: str = "http://localhost:11434",
        chroma_path: str = "./data/chroma_db",
    ):
        self.model       = model
        self.ollama_host = ollama_host
        self.chroma_path = chroma_path
        self.model_cfg   = SUPPORTED_MODELS.get(model, SUPPORTED_MODELS[DEFAULT_MODEL])

        logger.info(f"LLM Generator initialized")
        logger.info(f"  Model:  {model}")
        logger.info(f"  Desc:   {self.model_cfg['description']}")
        logger.info(f"  Temp:   {self.model_cfg['temperature']}")

        self._init_rag()

    # ── Model helpers ──────────────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        """Return model-appropriate system prompt."""
        if "deepseek" in self.model or "qwen" in self.model:
            return BMS_SYSTEM_PROMPT_DEEPSEEK
        return BMS_SYSTEM_PROMPT

    def _get_options(self, override_temp: Optional[float] = None) -> dict:
        """Build Ollama options dict for this model."""
        return {
            "temperature": override_temp if override_temp is not None else self.model_cfg["temperature"],
            "num_predict": self.model_cfg["num_predict"],
        }

    def _clean_response(self, raw: str) -> str:
        """
        Strip thinking tags and markdown from model response.
        Handles DeepSeek <think>...</think> and ```json ... ``` blocks.
        """
        # Remove DeepSeek/Qwen thinking blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        # Remove markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        return raw.strip()

    # ── RAG ───────────────────────────────────────────────────────────────────

    def _init_rag(self):
        if not CHROMA_AVAILABLE:
            self.collection = None
            return
        try:
            client = chromadb.PersistentClient(path=self.chroma_path)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.collection = client.get_or_create_collection(
                name="bms_requirements", embedding_function=ef,
            )
            logger.info(f"ChromaDB initialized — {self.collection.count()} docs")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            self.collection = None

    def index_requirements(self, requirements: list[dict]):
        if not self.collection:
            return
        documents, ids, metadatas = [], [], []
        for req in requirements:
            text   = f"{req.get('text_en','')} {req.get('text_de','')}"
            doc_id = str(req.get("Item ID", ""))
            if text.strip() and doc_id:
                documents.append(text)
                ids.append(doc_id)
                metadatas.append({
                    "section": str(req.get("Gliederungsnummer", "")),
                    "status":  req.get("Status", ""),
                })
        if documents:
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            logger.info(f"Indexed {len(documents)} requirements into ChromaDB")

    def get_rag_context(self, query: str, n_results: int = 3) -> str:
        if not self.collection or self.collection.count() == 0:
            return ""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            docs = results.get("documents", [[]])[0]
            return "\n\nSIMILAR REQUIREMENTS FOR CONTEXT:\n" + "\n---\n".join(docs[:3])
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return ""

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            models   = ollama.list()
            names    = [m["name"] for m in models.get("models", [])]
            # Check exact match or prefix match (cloud models have tags)
            return any(
                self.model == n or self.model in n or n in self.model
                for n in names
            )
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """Return all models available in the Ollama instance."""
        if not OLLAMA_AVAILABLE:
            return []
        try:
            return [m["name"] for m in ollama.list().get("models", [])]
        except Exception:
            return []

    # ── Main Generation ───────────────────────────────────────────────────────

    def generate_test_cases(
        self,
        requirement: dict,
        temperature: Optional[float] = None,
        n_tcs: int = 5,
    ) -> list[dict]:
        """
        Generate enhanced bilingual test cases for a BMS requirement.

        Args:
            requirement: enriched requirement dict from NLP pipeline
            temperature: override model default temperature
            n_tcs: number of test cases to request (3-7 recommended)

        Returns:
            List of test case dicts, empty list if generation fails.
        """
        if not self.is_available():
            logger.warning(f"Model {self.model} not available — skipping LLM generation")
            logger.warning(f"Available models: {self.list_available_models()}")
            return []

        section     = requirement.get("Gliederungsnummer", "")
        text_en     = requirement.get("text_en", "")
        text_de     = requirement.get("text_de", "")
        entities    = requirement.get("bms_entities", {})
        if isinstance(entities, str):
            try:    entities = json.loads(entities)
            except: entities = {}
        criticality = requirement.get("criticality_score", 0)
        is_crit     = requirement.get("is_critical", False)
        topic_en    = requirement.get("topic_label_en", "")
        ecu_level   = requirement.get("ecu_level", "")

        rag_context = self.get_rag_context(f"{text_en} {text_de}")

        # Build dynamic user prompt
        crit_note = (
            "\n⚠️  SAFETY-CRITICAL REQUIREMENT (Score: {}/10) — "
            "Include at least one FAULT test case with Priority=Critical.\n".format(criticality)
            if is_crit else ""
        )

        prompt = f"""{FEW_SHOT_EXAMPLE}

NOW GENERATE TEST CASES FOR THE FOLLOWING BMS REQUIREMENT:
{crit_note}
Section:           {section}
Topic:             {topic_en}
ECU Level:         {ecu_level}
Criticality Score: {criticality}
Signals:           {', '.join(entities.get('signals', [])[:5]) or 'N/A'}
Thresholds:        {', '.join(entities.get('thresholds', []))} °C
Qualifiers:        {', '.join(entities.get('qualifiers', [])) or 'N/A'}
Error Values:      {', '.join(entities.get('error_vals', [])) or 'N/A'}

🇩🇪 GERMAN REQUIREMENT (primary source):
{text_de}

🇬🇧 ENGLISH REQUIREMENT:
{text_en}
{rag_context}

Generate {n_tcs} test cases covering: Nominal, Boundary (one per threshold), Fault, Out-of-Range.
Output ONLY a valid JSON array. No markdown. No explanation."""

        logger.info(f"Generating TCs for section {section} using {self.model}...")

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user",   "content": prompt},
                ],
                options=self._get_options(temperature),
            )

            raw = response["message"]["content"]
            raw = self._clean_response(raw)

            # Extract JSON array robustly
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                # Try to find first { ... } pattern and wrap it
                obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if obj_match:
                    raw = f"[{obj_match.group()}]"
                    json_match = re.search(r"\[.*\]", raw, re.DOTALL)

            if json_match:
                tcs = json.loads(json_match.group())

                # Validate and tag each TC
                valid_tcs = []
                required_fields = ["TC_ID","Type_EN","Objective_EN","Ziel_DE",
                                   "Expected_Result_EN","Priority"]
                for i, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        continue
                    # Fill missing required fields with defaults
                    if "TC_ID" not in tc:
                        tc["TC_ID"] = f"TC_{str(section).replace('.','_')}_{100+i:03d}"
                    if "Type_DE" not in tc:
                        type_map = {"Nominal":"Normalfall","Boundary":"Grenzwert",
                                    "Fault":"Fehlerfall","Out-of-Range":"Bereichsüberschreitung"}
                        tc["Type_DE"] = type_map.get(tc.get("Type_EN",""), tc.get("Type_EN",""))
                    if "Priority" not in tc:
                        tc["Priority"] = "Critical" if is_crit else "High"
                    # Add metadata
                    tc["Requirement_ID"] = requirement.get("Item ID", "")
                    tc["Section"]        = section
                    tc["Category"]       = "LLM-Generated"
                    tc["Source"]         = self.model
                    valid_tcs.append(tc)

                logger.success(f"✅ {self.model} generated {len(valid_tcs)} TCs for section {section}")
                return valid_tcs

            else:
                logger.warning(f"Could not parse JSON from {self.model} response for section {section}")
                logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for section {section}: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM generation failed for section {section}: {e}")
            return []

    # ── Batch Generation ──────────────────────────────────────────────────────

    def generate_all(
        self,
        requirements: list[dict],
        temperature: Optional[float] = None,
        only_critical: bool = False,
    ) -> list[dict]:
        """
        Generate LLM test cases for all (or only critical) requirements.

        Args:
            requirements: list of enriched requirement dicts
            temperature: override temperature
            only_critical: if True, only process requirements with is_critical=True

        Returns:
            Flat list of all generated test cases.
        """
        all_tcs = []
        to_process = [
            r for r in requirements
            if (not only_critical) or r.get("is_critical", False)
        ]
        total = len(to_process)
        logger.info(f"Generating LLM TCs for {total} requirements using {self.model}")

        for i, req in enumerate(to_process, 1):
            section = req.get("Gliederungsnummer", "?")
            logger.info(f"  [{i}/{total}] Section {section}")
            tcs = self.generate_test_cases(req, temperature=temperature)
            all_tcs.extend(tcs)

        logger.success(f"✅ Total LLM TCs generated: {len(all_tcs)}")
        return all_tcs

    # ── RAG Chat ──────────────────────────────────────────────────────────────

    def answer_question(self, question: str, context_df=None) -> str:
        """
        Answer a natural language question about BMS requirements.
        Uses RAG + LLM. Supports DE and EN questions.
        """
        if not self.is_available():
            return (
                f"⚠️ Model `{self.model}` not available.\n"
                f"Available models: {', '.join(self.list_available_models()) or 'none'}\n"
                f"Run: `ollama serve`"
            )

        rag_ctx  = self.get_rag_context(question, n_results=5)
        df_ctx   = ""
        if context_df is not None:
            try:
                sample = context_df[["Gliederungsnummer","text_en","criticality_score"]].head(10)
                df_ctx = f"\n\nREQUIREMENT DATASET SAMPLE:\n{sample.to_string()}"
            except Exception:
                pass

        prompt = f"""You are a BMS (Battery Management System) validation expert assistant.
Answer the following question about BMS requirements clearly and concisely.
Reference specific requirement sections when possible.
{rag_ctx}
{df_ctx}

QUESTION: {question}

Answer in the same language as the question (DE or EN)."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self._get_options(override_temp=0.3),
            )
            raw = response["message"]["content"]
            return self._clean_response(raw)
        except Exception as e:
            return f"❌ LLM error ({self.model}): {str(e)}"

    # ── Model Info ────────────────────────────────────────────────────────────

    def model_info(self) -> dict:
        """Return information about the current model configuration."""
        return {
            "model":       self.model,
            "description": self.model_cfg.get("description", ""),
            "temperature": self.model_cfg.get("temperature", 0.2),
            "num_predict": self.model_cfg.get("num_predict", 2000),
            "recommended": self.model_cfg.get("recommended_for", ""),
            "available":   self.is_available(),
            "all_models":  self.list_available_models(),
        }
