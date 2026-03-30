# 🔋 BMS GenAI Assistant

> **Automated Bilingual Test Case Generation for BMS ECU Requirements**  
> Local · Open-Source LLM (Mistral/LLaMA) · spaCy DE+EN · ECU.TEST API · CI/CD

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20Mistral-purple)](https://ollama.ai)
[![ECU.TEST](https://img.shields.io/badge/ECU.TEST-2026.1-orange)](https://www.tracetronic.com)

---

## 📋 Project Context

This tool was developed as part of an internship project at a BMS ECU validation team.
It automates the creation of test cases from BMS functional specifications, reducing
test development time by ~92% while improving coverage and consistency.

**Specification:** `docs/CDC_BMS.pdf` — GenAI Assistant for BMS ECU Test Case Generation

---

## 🏗️ Architecture (7 Layers)

```
📥 INPUT          Excel/PDF requirements (HVSCC format, bilingual DE+EN)
    ↓
🧠 NLP            spaCy DE+EN · TF-IDF · LDA Topics · K-Means · Criticality Scoring
    ↓
🤖 LLM            Ollama Mistral-7B (local) · ChromaDB RAG · JSON output
    ↓
🧪 TC GENERATION  4 Unit types + 6 ECU Integration types · fully bilingual
    ↓
🔌 ECU.TEST       REST API (execution) · Object API (.pkg/.prj) · XML export
    ↓
⚙️ CI/CD          GitLab Runner (local) · 7 stages · auto-trigger on file drop
    ↓
🖥️ DASHBOARD      Streamlit (11 pages) · edit/approve · chat · metrics · export
```

---

## 🚀 Quick Start

### Option A — Manual (Development)

```bash
# 1. Setup
py -3.12 -m venv bms_env && bms_env\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# 2. Start backend (Terminal 1)
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 3. Start dashboard (Terminal 2)
streamlit run frontend/dashboard.py

# 4. Open http://localhost:8501
#    Login: engineer / bms2024
```

### Option B — Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
docker exec bms_ollama ollama pull mistral  # one-time ~4GB
# Open http://localhost:8501
```

### Option C — Pipeline Only (No UI)

```bash
python cicd/run_pipeline.py data/uploads/HVSCC-38653_AM_Final.xlsx
python cicd/run_pipeline.py data/uploads/HVSCC-38653_AM_Final.xlsx --llm
```

---

## 📁 Project Structure

```
bms_project/
│
├── backend/
│   ├── api.py              ← FastAPI REST backend (11 endpoints)
│   ├── database.py         ← SQLite + SQLAlchemy ORM
│   └── ingestion.py        ← Excel/PDF requirements loader + validator
│
├── nlp/
│   ├── pipeline.py         ← BMSNLPPipeline (spaCy DE+EN, TF-IDF, LDA, criticality)
│   └── test_generator.py   ← BMSTestCaseGenerator (4 unit + 6 ECU types)
│
├── llm/
│   └── generator.py        ← Ollama Mistral + ChromaDB RAG + prompt engineering
│
├── ecu/
│   └── ecutest_integration.py ← ECU.TEST REST API + Object API + XML export
│
├── frontend/
│   ├── dashboard.py        ← Streamlit dashboard (11 pages)
│   └── .streamlit/config.toml
│
├── cicd/
│   ├── .gitlab-ci.yml      ← GitLab CI/CD (7 stages)
│   ├── run_pipeline.py     ← Local pipeline runner
│   ├── watcher.py          ← File system auto-trigger
│   └── generate_report.py  ← HTML coverage report
│
├── docs/
│   ├── 01_Architecture.md          ← System architecture document
│   ├── 02_NLP_AI_Pipeline.md       ← NLP/AI pipeline documentation
│   ├── 03_Prompt_Engineering.md    ← Prompt engineering methodology
│   ├── 04_Validation_Evaluation.md ← TC evaluation & metrics
│   └── 05_User_Guide.md            ← Engineer user guide
│
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
│
├── tests/
│   └── test_pipeline.py    ← 30+ pytest tests
│
├── data/uploads/           ← Drop .xlsx files here
├── outputs/                ← Generated TCs, reports, ECU.TEST packages
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## 🎯 Objectives Met (from CDC_BMS.pdf)

| Objective | Status | Implementation |
|-----------|--------|----------------|
| Requirements Understanding | ✅ Full | spaCy DE+EN, entity extraction, TF-IDF, LDA |
| Automatic TC Generation | ✅ Full | 4 unit types + 6 ECU types, bilingual |
| Critical Scenario Identification | ✅ Full | 10 pattern categories, criticality score |
| Test Script Creation | ✅ Full | ECU.TEST REST + Object API + XML |
| Quality Analysis & Consistency | ✅ Full | Ambiguity, missing units, EN↔DE check |
| User Interaction Module | ✅ Full | 11-page dashboard, edit, approve, chat |

---

## 🧪 Test Case Types Generated

### Unit Tests (SIL)
| Type | Trigger |
|------|---------|
| Nominal / Normalfall | Every requirement |
| Boundary / Grenzwert | When temperature thresholds found |
| Fault / Fehlerfall | When qualifier/plausibility keywords found |
| Out-of-Range / Bereichsüberschreitung | When ≥1 threshold present |

### ECU Integration Tests (HIL)
| Type | Tool | Trigger |
|------|------|---------|
| End-to-End Signal Flow | HIL + dSPACE | Every requirement |
| OBD / Diagnostic Interface | ISTA + CANoe DiagVistA | OBD/PID/DTC keywords |
| CAN Bus Signal Verification | CANalyzer / CANoe | Signal names extracted |
| HIL Fault Injection | dSPACE / NI VeriStand | Critical requirements |
| Timing & Cycle Verification | CANoe + Oscilloscope | ms/cycle keywords |
| Error Memory / DTC | ISTA + python-udsoncan | DTC/Fehlerspeicher keywords |

---

## 🔌 ECU.TEST API Integration

Uses official Tracetronic APIs (v2026.1):

```python
# REST API — execute test packages
from ecu.ecutest_integration import ECUTestRESTClient
client = ECUTestRESTClient()
client.load_configuration("MySetup.tbc", "BMS.tcf")
result = client.execute_package("TC_17_8_001.pkg")

# Object API — create .pkg files programmatically
from ecu.ecutest_integration import ECUTestPackageGenerator
gen = ECUTestPackageGenerator()
gen.create_package_from_tc(tc_dict, output_dir)

# XML Export (no ECU.TEST install needed)
from ecu.ecutest_integration import ECUTestExporter
exp = ECUTestExporter()
exp.export_project_xml(df_unit, df_ecu, output_dir)
```

---

## 🤖 LLM Integration

```python
from llm.generator import BMSLLMGenerator

llm = BMSLLMGenerator(model="mistral")

# Index requirements for RAG
llm.index_requirements(df.to_dict("records"))

# Generate enhanced TCs
tcs = llm.generate_test_cases(requirement_dict)

# Answer questions (RAG chat)
answer = llm.answer_question("Welche Anforderungen haben Temperaturgrenzwerte?")
```

**Start Ollama:**
```bash
ollama serve
ollama pull mistral   # or: ollama pull llama3
```

---

## 📊 Dashboard Pages (11)

| Page | Function |
|------|----------|
| 📊 Dashboard | KPIs, charts, topic distribution |
| 📥 Upload | File upload, pipeline trigger |
| 🧠 NLP Analysis | Topics, criticality, quality, EN↔DE check |
| 🧪 Unit Test Cases | Filter, view, approve unit TCs |
| 🔌 ECU Integration | View ECU TCs by type and level |
| 📝 Edit & Approve | **Inline TC editing** + approve/revoke |
| ➕ Manual TC Builder | Create custom TCs |
| 📈 Evaluation Metrics | Coverage %, per-req table, approval stats |
| 💬 Chat (RAG) | Natural language Q&A with Mistral |
| ⚙️ LLM Settings | Temperature, model, prompt customization |
| 📤 Export | Excel, ECU.TEST XML, coverage report |

---

## 🔬 Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## ⚙️ CI/CD Pipeline (7 Stages)

```
validate → nlp → llm_generate → ecu_tests → coverage → export → notify
```

**Local run:** `python cicd/run_pipeline.py <file.xlsx>`  
**Auto-watch:** `python cicd/watcher.py` (monitors `data/uploads/`)  
**GitLab CI:** `.gitlab-ci.yml` — runs on local GitLab Runner

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| `docs/01_Architecture.md` | Full system architecture, data flow, tech stack |
| `docs/02_NLP_AI_Pipeline.md` | NLP steps, algorithms, parameters |
| `docs/03_Prompt_Engineering.md` | Prompt design, few-shot, RAG, iteration history |
| `docs/04_Validation_Evaluation.md` | Coverage metrics, quality scores, evaluation framework |
| `docs/05_User_Guide.md` | Step-by-step engineer guide |
| `BMS_Architecture.html` | Interactive architecture diagram |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP | spaCy (`de_core_news_sm` + `en_core_web_sm`), scikit-learn, NLTK |
| LLM | Ollama + Mistral-7B / LLaMA-3 (fully local) |
| RAG | ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` |
| Backend | FastAPI + uvicorn + SQLAlchemy + SQLite |
| Frontend | Streamlit + Plotly |
| Auth | JWT (HMAC-SHA256) |
| ECU | ECU.TEST REST API + Object API (Tracetronic 2026.1) |
| CI/CD | GitLab CE + Docker |
| Export | openpyxl + xlsxwriter + Jinja2 |

---

## 🔐 Default Credentials

| User | Password | Role |
|------|----------|------|
| `engineer` | `bms2024` | Standard engineer |
| `admin` | `admin2024` | Administrator |

---

*BMS GenAI Assistant v1.0 — Internship Project*
