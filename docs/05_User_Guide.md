# BMS GenAI Assistant — User Guide

**Version:** 1.0 | **Audience:** Validation Engineers

---

## 1. Quick Start (5 Minutes)

### Prerequisites
- Python 3.12
- Git
- 4GB free disk space (for LLM model)

### Step 1 — Setup
```bash
# Clone/extract the project
cd bms_project

# Create virtual environment
py -3.12 -m venv bms_env
bms_env\Scripts\activate          # Windows
# source bms_env/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Step 2 — Start Backend API
```bash
# In terminal 1 (keep open)
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```
✅ API running at: http://localhost:8000  
📖 API docs at: http://localhost:8000/docs

### Step 3 — Start Dashboard
```bash
# In terminal 2 (keep open)
streamlit run frontend/dashboard.py
```
✅ Dashboard at: http://localhost:8501

### Step 4 — Login
- URL: http://localhost:8501
- Username: `engineer`
- Password: `bms2024`

### Step 5 — Upload Requirements
1. Go to **📥 Upload Requirements**
2. Upload `HVSCC-38653_AM_Final.xlsx`
3. Click **🚀 Run Full Pipeline**
4. Wait ~30 seconds
5. Navigate to **📊 Dashboard** to see results

---

## 2. Optional: Enable LLM (Mistral)

```bash
# Install Ollama: https://ollama.ai/download
# Then in a new terminal:
ollama serve              # Start Ollama server
ollama pull mistral       # Download Mistral-7B (~4GB, one-time)
```

In the Upload page, toggle **🤖 Enable LLM** before running the pipeline.

---

## 3. Dashboard Pages Guide

### 📊 Dashboard
**What it shows:** High-level KPIs after processing.
- Requirements count, critical count, generated TCs
- Topic distribution pie chart
- Criticality score histogram
- Word count comparison EN vs DE

---

### 📥 Upload Requirements
**What to do:**
1. Upload your `.xlsx` requirements file (HVSCC format)
2. Toggle LLM if Ollama is running
3. Click Run Pipeline
4. Review validation warnings if any

**Supported format:** HVSCC Excel with columns:
`Item ID`, `Gliederungsnummer`, `Beschreibung`, `Englisch`, `Typ`, `Status`

---

### 🧠 NLP Analysis
**5 tabs:**

| Tab | Content |
|-----|---------|
| 📋 Requirements | Full requirements table with NLP enrichment |
| 🗂️ Topics | LDA topic distribution (DE + EN) |
| 🔴 Criticality | Complexity vs criticality scatter, critical requirements list |
| ⚠️ Quality | Detected quality issues (vague terms, missing units) |
| 🔍 EN↔DE Check | Signal overlap and length ratio consistency scores |

**Tip:** Sort the EN↔DE Check table by Score (ascending) to find inconsistencies first.

---

### 🧪 Unit Test Cases
**What it shows:** All generated software unit test cases.

**Filters:**
- By Type: Nominal, Boundary, Fault, Out-of-Range
- By Priority: Critical, High, Medium, Low

**Approve TCs:**
- Click **✅ Approve ALL filtered** to approve all visible TCs
- Click **✅ Approve Critical only** for safety-relevant items

---

### 🔌 ECU Integration
**What it shows:** All generated ECU.TEST integration test cases.

**6 integration types:**
1. End-to-End Signal Flow (HIL)
2. OBD / Diagnostic Interface
3. CAN Bus Signal Verification
4. HIL Fault Injection
5. Timing & Cycle Verification
6. Error Memory / DTC

**Filter** by integration type to focus on specific test areas.

---

### 📝 Edit & Approve
**How to edit a TC:**
1. Select TC ID from dropdown
2. Edit any field in the text areas (EN or DE)
3. Click **💾 Save Edits**
4. Click **✅ Approve** to mark as approved

**Tip:** Focus on Critical priority TCs first — these have highest safety impact.

**Approve all:** Go to 🧪 Unit Test Cases page and click "Approve ALL filtered."

---

### ➕ Manual TC Builder
**When to use:**
- Regression test cases from past defects
- Exploratory testing scenarios
- Test cases not derivable from requirements (e.g., timing edge cases)

**Required fields:**
- TC ID (auto-generated but editable)
- Objective (EN + DE)
- Preconditions (EN + DE)
- Expected Result (EN + DE)
- Priority, Type, Test Environment

---

### 📈 Evaluation Metrics
**What it shows:**
- Requirement Coverage % (requirements with TCs / total)
- Critical Coverage % (critical reqs with TCs / total critical)
- TC Approval Rate (approved / total)
- Avg TCs per Requirement
- Per-requirement coverage table (shows uncovered requirements ❌)

**Action:** Review uncovered requirements (❌) and manually add TCs if needed.

---

### 💬 Chat (RAG + Mistral)
**How to use:**
- Type any question in English or German
- The system retrieves similar requirements from ChromaDB (RAG)
- Mistral answers using requirement context

**Example questions:**
```
"Which requirements are safety-critical?"
"What are the temperature thresholds for cell core temperature?"
"Welche Anforderungen haben Qualifier-Bedingungen?"
"Show me all OBD-related requirements."
```

**Note:** Requires Ollama running (`ollama serve`).

---

### ⚙️ LLM Settings
**Adjustable parameters:**
- **Model:** mistral (default) or llama3
- **Temperature:** 0.0-1.0 (default 0.2 — keep low for precise output)
- **Max Tokens:** 500-4000 (default 2000)
- **Focus Area:** All / Temperature / Fault Handling / OBD / Safety / Timing

**Prompt Customization:**
- Edit the System Prompt to adjust LLM behavior
- View the current few-shot example
- Configure RAG context documents count

---

### 📤 Export
**Available exports:**

| Format | Contents | Use |
|--------|----------|-----|
| Excel (.xlsx) | Unit TCs + ECU TCs + Manual TCs + Edited TCs + Summary | Engineer review, DOORS import |
| ECU.TEST XML | .pkg.xml + .prj.xml for all TCs | Import into ECU.TEST |
| Coverage Report | HTML dashboard with coverage metrics | Project reporting |

---

## 4. Running the Pipeline Without UI

```bash
# Activate environment
bms_env\Scripts\activate

# Run full pipeline on a file
python cicd/run_pipeline.py data/uploads/HVSCC-38653_AM_Final.xlsx

# With LLM enhancement
python cicd/run_pipeline.py data/uploads/HVSCC-38653_AM_Final.xlsx --llm

# Output in: ./outputs/
```

**Generated files:**
```
outputs/
├── nlp_HVSCC-38653_AM_Final.json          ← NLP-enriched requirements
├── unit_tcs_HVSCC-38653_AM_Final.xlsx     ← Unit test cases
├── ecu_tcs_HVSCC-38653_AM_Final.xlsx      ← ECU integration TCs
├── bms_testcases_HVSCC-38653_..._*.xlsx   ← Combined all sheets
├── ecutest_HVSCC-38653_AM_Final/          ← ECU.TEST packages
│   ├── TC_17_8_001.pkg.xml
│   ├── TC_17_8_010.pkg.xml
│   └── BMS_HVSCC-38653_AM_Final.prj.xml
└── coverage_report.html                   ← Coverage dashboard
```

---

## 5. Auto-Watch Mode

Drop any `.xlsx` file into `data/uploads/` and the pipeline runs automatically:

```bash
# Start file watcher (background)
python cicd/watcher.py
```

---

## 6. Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
# Open: htmlcov/index.html

# Quick smoke test
pytest tests/test_pipeline.py::TestIngestion -v
```

---

## 7. Importing ECU.TEST Packages

### If ECU.TEST is installed:
1. Open ECU.TEST
2. File → Import → Package (.pkg.xml)
3. Navigate to `outputs/ecutest_*/`
4. Select individual `.pkg.xml` files or the `.prj.xml` project

### Executing via REST API:
```python
from ecu.ecutest_integration import ECUTestRESTClient

client = ECUTestRESTClient(host="127.0.0.1", port=5050)
result = client.load_configuration("MySetup.tbc", "BMS_Config.tcf")
result = client.execute_package("outputs/ecutest/TC_17_8_001.pkg")
print(result["status"])  # PASSED / FAILED
```

---

## 8. Default Credentials

| User | Password | Role |
|------|----------|------|
| `engineer` | `bms2024` | View + Edit + Approve |
| `admin` | `admin2024` | Full access |

**Change passwords:** Edit `backend/api.py` → `USERS_DB` → update `_hash_password("newpassword")`.

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `API Offline` in dashboard | Run `uvicorn backend.api:app --port 8000` |
| `Ollama: 🔴` in sidebar | Run `ollama serve` in a terminal |
| `spaCy model not found` | Run `python -m spacy download de_core_news_sm` |
| Upload fails with 500 error | Check API terminal for error message |
| Plotly charts not showing | Add `pio.renderers.default = "notebook"` |
| `numpy.int64` serialization error | Already fixed in v1.1 — update `backend/api.py` |
| Port 8000 in use | `uvicorn backend.api:app --port 8001` + update `API_URL` in dashboard |

---

## 10. Architecture Diagram

See `BMS_Architecture.html` for the interactive architecture diagram.

For documentation:
- `docs/01_Architecture.md` — System architecture
- `docs/02_NLP_AI_Pipeline.md` — NLP/AI pipeline details
- `docs/03_Prompt_Engineering.md` — Prompt engineering methodology
- `docs/04_Validation_Evaluation.md` — Test case evaluation metrics

---

*User Guide — BMS GenAI Assistant v1.0*
