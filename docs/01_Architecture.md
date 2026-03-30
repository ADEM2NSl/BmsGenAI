# BMS GenAI Assistant — Technical Architecture Document

**Project:** GenAI Assistant for Automatic Test Case Generation for a BMS ECU  
**Version:** 1.0  
**Date:** 2025  

---

## 1. Executive Summary

The BMS GenAI Assistant is a locally-deployed intelligent tool that automatically
generates exhaustive, bilingual (German + English) test cases for Battery Management
System (BMS) ECU validation. It combines Natural Language Processing (NLP), a local
Large Language Model (LLM), and the Tracetronic ECU.TEST API to cover the full
validation workflow — from requirement ingestion to test script execution.

**Key Design Decisions:**
- **100% local deployment** — no cloud, no external API keys, protecting sensitive automotive data
- **Open-source LLM** — Ollama + Mistral-7B / LLaMA-3 running on the engineer's machine
- **Bilingual DE+EN** — requirements are in German (primary), English (secondary); all outputs in both
- **ECU.TEST integration** — uses official Tracetronic REST + Object API for test execution

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BMS GenAI Assistant                           │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐             │
│  │  INPUT   │→  │  NLP ENGINE  │→  │  LLM ENGINE│             │
│  │  LAYER   │   │  (spaCy      │   │  (Ollama + │             │
│  │          │   │   DE + EN)   │   │   Mistral) │             │
│  └──────────┘   └──────────────┘   └────────────┘             │
│        │                │                  │                   │
│        ▼                ▼                  ▼                   │
│  ┌──────────────────────────────────────────────┐             │
│  │              TC GENERATION ENGINE             │             │
│  │   Unit TCs (4 types) + ECU TCs (6 types)     │             │
│  └──────────────────────────────────────────────┘             │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐             │
│  │  CI/CD   │   │  WEB DASH-  │   │  ECU.TEST  │             │
│  │ PIPELINE │   │  BOARD      │   │  EXPORT    │             │
│  │(7 stages)│   │ (Streamlit) │   │(REST+ObjAPI│             │
│  └──────────┘   └──────────────┘   └────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-by-Layer Architecture

### 3.1 Input Layer (`backend/ingestion.py`)

**Purpose:** Load and validate BMS requirement files.

**Supported formats:**
| Format | Library | Notes |
|--------|---------|-------|
| Excel (.xlsx) | openpyxl + pandas | HVSCC format, skiprows=3 |
| PDF | pypdf | Specification documents |

**Validation checks (pydantic):**
- Required columns present: `Item ID`, `Gliederungsnummer`, `Beschreibung`, `Englisch`, `Typ`, `Status`
- Only `Anforderung` (requirement) rows processed; headings filtered out
- Missing text detection, duplicate section numbers, unknown status values

**File watcher (`cicd/watcher.py`):**
- Uses `watchdog` library to monitor `./data/uploads/`
- Auto-triggers the full pipeline when a new `.xlsx` is dropped

---

### 3.2 NLP Pipeline (`nlp/pipeline.py`)

**Class:** `BMSNLPPipeline`

**Processing steps:**

```
Raw Text (DE + EN)
      │
      ▼
┌─────────────────────────────────┐
│  1. BMS Text Cleaning           │
│     - Remove BMW_* signal names │
│     - Remove QUAL_* qualifiers  │
│     - Remove numbers/units      │
│     - Remove (Unit:...) tags    │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  2. spaCy Lemmatization         │
│     DE: de_core_news_sm         │
│     EN: en_core_web_sm          │
│     - Tokenize → Lemmatize      │
│     - Remove stopwords (DE+EN)  │
│     - Remove short tokens (<3)  │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  3. BMS Entity Extraction       │
│     Regex patterns:             │
│     - BMW_[A-Za-z0-9_]+        │  → signals
│     - (-?\d+\.?\d*)\s*°C       │  → thresholds
│     - QUAL_[A-Z_]+             │  → qualifiers
│     - BMW_LIM_[A-Za-z0-9_]+   │  → error values
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  4. TF-IDF Vectorization        │
│     max_features=200, min_df=1  │
│     ngram_range=(1,1)           │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  5. LDA Topic Modeling          │
│     n_topics=5, max_iter=20     │
│     Run separately DE and EN    │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  6. K-Means Clustering          │
│     n_clusters=4                │
│     Bilingual TF-IDF (DE+EN)   │
│     TruncatedSVD for 2D viz     │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  7. Criticality Scoring         │
│     10 pattern categories       │
│     Score = sum of matches      │
│     is_critical = score >= 3    │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  8. Quality Analysis            │
│     6 issue categories DE+EN    │
│     Vague terms, missing units  │
└─────────────────────────────────┘
```

**Criticality patterns (10 categories):**

| Category | Language | Examples |
|----------|----------|---------|
| range_check | EN | range, boundary, limit, min, max |
| fault | EN | fault, error, failure, invalid |
| safety | EN | safety, critical, shutdown |
| timing | EN | timeout, debouncing, cycle |
| plausibility | EN | plausib, qual_int, qualifier |
| bereich | DE | bereich, grenze, grenzwert |
| fehler | DE | fehler, defekt, ungültig |
| sicherheit | DE | sicherheit, kritisch, schutz |
| plausibilitaet | DE | plausib, n.i.o |
| bedingung | DE | wenn, falls, sobald |

**Topic model labels (5 topics per language):**

| Topic | EN Label | DE Label |
|-------|----------|----------|
| 1 | Temperature Monitoring | Temperaturüberwachung |
| 2 | Fault & Plausibility | Fehler & Plausibilität |
| 3 | Signal Output | Signalausgabe |
| 4 | OBD Data | OBD-Daten |
| 5 | Range & Conditions | Bereich & Bedingungen |

---

### 3.3 LLM Engine (`llm/generator.py`)

**Class:** `BMSLLMGenerator`

**Architecture:**

```
Requirement (DE + EN)
      │
      ▼
┌─────────────────────────────┐
│  RAG: ChromaDB Query        │
│  - paraphrase-multilingual  │
│    MiniLM-L12-v2 embeddings │
│  - Top-3 similar reqs       │
│    as context               │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Prompt Construction        │
│  - System prompt (BMS expert│
│  - Few-shot example         │
│  - Requirement context      │
│  - Extracted entities       │
│  - RAG context              │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Ollama API Call            │
│  model: mistral/llama3      │
│  temperature: 0.2           │
│  format: JSON array         │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Output Parser              │
│  - Extract JSON from text   │
│  - Validate TC structure    │
│  - Add metadata             │
└─────────────────────────────┘
```

**Ollama API endpoint:**
```
POST http://localhost:11434/api/chat
{
  "model": "mistral",
  "messages": [
    {"role": "system", "content": BMS_SYSTEM_PROMPT},
    {"role": "user",   "content": requirement_prompt}
  ],
  "options": {"temperature": 0.2, "num_predict": 2000}
}
```

---

### 3.4 Test Case Generator (`nlp/test_generator.py`)

**Class:** `BMSTestCaseGenerator`

**Unit test case types (4):**

| Type (EN) | Type (DE) | Trigger Condition |
|-----------|-----------|-------------------|
| Nominal | Normalfall | Always generated |
| Boundary | Grenzwert | When thresholds extracted |
| Fault | Fehlerfall | When plausibility keywords found |
| Out-of-Range | Bereichsüberschreitung | When ≥1 threshold present |

**ECU integration test case types (6):**

| Type | Environment | Trigger |
|------|-------------|---------|
| End-to-End Signal Flow | HIL | Always |
| OBD / Diagnostic Interface | HIL + ISTA | OBD/PID/DTC keywords |
| CAN Bus Signal Verification | HIL + CANoe | Signals extracted |
| HIL Fault Injection | dSPACE / NI VeriStand | Critical or fault keywords |
| Timing & Cycle Verification | CANoe + Oscilloscope | ms/cycle keywords |
| Error Memory / DTC | HIL + DiagVistA | DTC/Fehlerspeicher keywords |

**Bilingual fields in every TC:**
- `Objective_EN` / `Ziel_DE`
- `Preconditions_EN` / `Vorbedingungen_DE`
- `Expected_Result_EN` / `Erwartetes_Ergebnis_DE`
- `Type_EN` / `Type_DE`

---

### 3.5 ECU.TEST Integration (`ecu/ecutest_integration.py`)

**Three API layers used:**

#### REST API (`ECUTestRESTClient`)
- Base URL: `http://127.0.0.1:5050/api/v2`
- `PUT /configuration` → load TBC + TCF
- `PUT /execution` → run package
- `GET /reports/{id}` → get results
- `PUT /reports/{id}/upload` → push to test.guide
- Polling: `WaitForOperationEnd()` checks status every 1s

#### Object API (`ECUTestPackageGenerator`)
- Requires ECU.TEST installed: `C:\Program Files\ecu.test\Templates\ApiClient`
- `PackageApi.CreatePackage()` → create `.pkg` file
- Test steps: `TsWrite` (set signals), `TsWait` (delay), `TsRead` (verify with `expectationExpression`)
- `ProjectApi.CreateProject()` → group packages into `.prj`

#### XML Export Fallback (`ECUTestExporter`)
- Works without ECU.TEST installed
- Generates `.pkg.xml` and `.prj.xml` in ECU.TEST import format
- Includes: `TsWrite`, `TsWait`, `TsRead` steps with `expectationExpression`
- Ready to import when ECU.TEST is available

---

### 3.6 FastAPI Backend (`backend/api.py`)

**Base URL:** `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/token` | POST | Login, returns JWT |
| `/requirements/upload` | POST | Upload Excel, run NLP |
| `/requirements/{id}` | GET | Get enriched requirements |
| `/requirements/{id}/stats` | GET | NLP statistics |
| `/testcases/generate/{id}` | POST | Generate all TCs |
| `/testcases/{id}` | GET | Get test cases |
| `/testcases/{id}/approve` | POST | Approve selected TCs |
| `/export/{id}/excel` | GET | Download Excel |
| `/export/{id}/canoe_xml` | GET | Download ECU.TEST XML |
| `/chat` | POST | RAG chat query |
| `/health` | GET | System health |

**Authentication:** JWT (HMAC-SHA256, no bcrypt dependency)  
**Database:** SQLite + SQLAlchemy ORM  
**Numpy serialization:** Custom `numpy_safe()` converter for all responses  

---

### 3.7 CI/CD Pipeline (`.gitlab-ci.yml`)

**7 automated stages:**

```
PUSH to Git
    │
    ▼ Stage 1: validate
    Schema check, encoding, mandatory columns
    │
    ▼ Stage 2: nlp
    spaCy DE+EN, TF-IDF, LDA, criticality
    │
    ▼ Stage 3: llm_generate
    Ollama Mistral (if available)
    │
    ▼ Stage 4: ecu_tests
    Validate ECU TC structure, export XML
    │
    ▼ Stage 5: coverage
    Generate HTML coverage report
    │
    ▼ Stage 6: export
    Package all artifacts into release folder
    │
    ▼ Stage 7: notify
    Email / Teams webhook notification
```

**Local runner:** GitLab Runner (no cloud needed)  
**Alternative:** `python cicd/run_pipeline.py <file.xlsx>` for direct local execution  

---

### 3.8 Web Dashboard (`frontend/dashboard.py`)

**11 pages:**

| Page | Function |
|------|----------|
| 📊 Dashboard | KPIs, topic chart, criticality histogram, word count |
| 📥 Upload | File upload, pipeline trigger, validation feedback |
| 🧠 NLP Analysis | Requirements table, topics, criticality scatter, quality issues, EN↔DE check |
| 🧪 Unit Test Cases | Filter, view, approve unit TCs |
| 🔌 ECU Integration | View ECU TCs by type and level |
| 📝 Edit & Approve | Inline TC editing, approve/revoke |
| ➕ Manual TC Builder | Create custom TCs not from requirements |
| 📈 Evaluation Metrics | Coverage %, per-req table, approval stats |
| 💬 Chat (RAG) | Natural language Q&A with Mistral + ChromaDB |
| ⚙️ LLM Settings | Temperature, model, focus area, prompt customization |
| 📤 Export | Excel (multi-sheet), ECU.TEST XML, coverage report |

---

## 4. Data Flow Diagram

```
.xlsx file
    │
    ▼
BMSRequirementLoader.load_excel()
    │ DataFrame (50 requirements)
    ▼
BMSNLPPipeline.run()
    │ Enriched DataFrame (+30 columns)
    ├── tokens_en, tokens_de (lemmatized)
    ├── bms_entities (signals, thresholds, qualifiers)
    ├── criticality_score, is_critical
    ├── topic_label_en, topic_label_de
    ├── cluster_label
    ├── ecu_level
    └── quality_issues
    │
    ▼
BMSDatabase.save_requirements()  →  SQLite (bms.db)
    │
    ├──► ChromaDB.index_requirements()  →  Vector store
    │
    ▼
BMSTestCaseGenerator.run()
    ├── df_unit  (~200 unit TCs)
    └── df_ecu   (~300 ECU integration TCs)
    │
    ├──► BMSLLMGenerator.generate_test_cases()  →  Enhanced TCs (optional)
    │
    ▼
BMSDatabase.save_test_cases()  →  SQLite
    │
    ├──► export_excel()        →  bms_testcases_*.xlsx
    └──► export_canoe_xml()    →  ECU.TEST .pkg.xml + .prj.xml
```

---

## 5. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **NLP** | spaCy | ≥3.7 | DE+EN lemmatization, NER |
| **NLP** | scikit-learn | ≥1.3 | TF-IDF, LDA, K-Means, SVD |
| **NLP** | NLTK | ≥3.8 | Stopwords (DE+EN) |
| **LLM** | Ollama | ≥0.1 | Local LLM runtime |
| **LLM** | Mistral-7B / LLaMA-3 | latest | Test case generation |
| **RAG** | ChromaDB | ≥0.4 | Vector store |
| **RAG** | sentence-transformers | ≥2.2 | Multilingual embeddings |
| **Backend** | FastAPI | ≥0.104 | REST API |
| **Backend** | SQLAlchemy + SQLite | ≥2.0 | Persistent storage |
| **Backend** | python-jose | ≥3.3 | JWT authentication |
| **Frontend** | Streamlit | ≥1.28 | Web dashboard |
| **ECU** | ECU.TEST REST API | 2026.1 | Test execution |
| **ECU** | ECU.TEST Object API | 2026.1 | Package creation |
| **CI/CD** | GitLab CE | local | Pipeline orchestration |
| **CI/CD** | Docker | latest | Containerization |
| **Monitoring** | Jinja2 | ≥3.1 | HTML report generation |

---

## 6. Deployment Architecture

```
Local Machine (Windows / Linux)
├── Ollama Service (port 11434)  ←── Mistral-7B model
├── FastAPI Backend (port 8000)  ←── bms.db, chroma_db
├── Streamlit Dashboard (port 8501)
└── File Watcher (background process)
         └── watches: ./data/uploads/
             triggers: python cicd/run_pipeline.py
```

**Docker Compose alternative:**
```yaml
services:
  api:       port 8000  (FastAPI + NLP)
  dashboard: port 8501  (Streamlit)
  ollama:    port 11434 (Mistral-7B)
  watcher:   background (file trigger)
```

---

## 7. Security Considerations

- All data stays on-premise — no external API calls (except optional model downloads)
- JWT authentication with configurable expiry (default 8h)
- Passwords hashed with HMAC-SHA256 (no bcrypt dependency issues)
- SQLite database stored locally in `./data/bms.db`
- `.env` file for secrets — never committed to Git

---

## 8. Performance Characteristics

| Operation | Approximate Time | Notes |
|-----------|-----------------|-------|
| Excel load + validation | <1s | 50 requirements |
| NLP pipeline (full) | 10-30s | spaCy DE+EN, LDA, clustering |
| Unit TC generation | <1s | Pure Python, no ML |
| ECU TC generation | <1s | Pure Python, no ML |
| LLM TC generation | 30-120s | Depends on hardware (CPU vs GPU) |
| Excel export | <1s | xlsxwriter |
| ECU.TEST XML export | 2-5s | File I/O for all packages |
| ChromaDB indexing | 5-15s | sentence-transformers embedding |

**GPU acceleration:** Set `OLLAMA_NUM_GPU=1` — reduces LLM time to ~10-30s  
**CPU mode:** Works on any machine — slower but fully functional

---

*Document generated by BMS GenAI Assistant — Architecture v1.0*
