# BMS GenAI Assistant — NLP/AI Pipeline Documentation

**Version:** 1.0 | **Language:** DE+EN Bilingual Pipeline

---

## 1. Pipeline Overview

The NLP/AI pipeline transforms raw BMS requirement text into structured, enriched data
used for test case generation. It operates in two phases:

1. **NLP Phase** — rule-based + statistical (always runs, fast, no GPU needed)
2. **LLM Phase** — generative (optional, requires Ollama, slower)

```
Phase 1: NLP (deterministic, <30s)
┌──────────────────────────────────────────────────────────────┐
│  Text Cleaning → Lemmatization → Entity Extraction           │
│  → TF-IDF → LDA Topic Modeling → K-Means Clustering         │
│  → Criticality Scoring → Quality Analysis                    │
└──────────────────────────────────────────────────────────────┘

Phase 2: LLM (generative, optional, 30-120s)
┌──────────────────────────────────────────────────────────────┐
│  RAG Query (ChromaDB) → Prompt Construction                  │
│  → Ollama/Mistral Inference → JSON Parsing → TC Assembly    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. NLP Phase — Step-by-Step

### Step 1: Text Cleaning (`BMSNLPPipeline.clean_bms()`)

BMS requirement text contains highly technical content that would confuse standard NLP:

**Removed elements:**
```python
re.sub(r'BMW_\w+', '', text)          # BMW signal names: BMW_t_HvsCellMinObd
re.sub(r'QUAL_[A-Z_]+', '', text)     # Qualifiers: QUAL_INT_OK
re.sub(r'-?\d+\.?\d*\s*°?[cCkK%]?', '', text)  # Numbers: -40, 215°C
re.sub(r'\(Unit:[^)]*\)', '', text)   # Unit annotations: (Unit: °C, Dim: 1x1)
re.sub(r'\(Dim:[^)]*\)', '', text)    # Dim annotations
```

**Why:** These tokens would dominate TF-IDF and topic models without adding semantic value.
Signal names vary per requirement but appear consistently — removing them forces the model
to focus on *what the requirement does*, not *which signal*.

**Example:**
```
Before: "Die BMU HL muss BMW_t_HvsCellMinObd (Unit: °C) ausgeben wenn QUAL_INT_OK"
After:  "Die BMU HL muss  ausgeben wenn "
```

### Step 2: spaCy Lemmatization (Bilingual)

**German (`de_core_news_sm`):**
```python
doc = nlp_de(text.lower())
tokens = [t.lemma_ for t in doc
          if t.is_alpha          # letters only
          and len(t.text) > 2    # min 3 chars
          and t.text not in STOP_DE   # custom stopwords
          and not t.is_stop]     # spaCy stopwords
```

**English (`en_core_web_sm`):**
```python
doc = nlp_en(text.lower())
tokens = [t.lemma_ for t in doc
          if t.is_alpha and len(t.text) > 2
          and t.text not in STOP_EN
          and not t.is_stop]
```

**German-specific challenges:**
- Compound words: `Temperaturüberwachung` → lemma `temperaturüberwachung` (kept as one token)
- Inflection: `ausgeben`, `ausgegeben`, `Ausgabe` → all → `ausgabe`
- Umlauts handled natively by spaCy DE model

**Custom stopwords added (beyond NLTK defaults):**

| Language | Added stopwords |
|----------|----------------|
| DE | muss, wird, soll, kann, bzw, sowie, gem, wert, ausgeben, bmw, bmu |
| EN | unit, dim, 1x1, must, bmw, bmu, hl, rc, equal, equals, output, value, input |

### Step 3: BMS Entity Extraction (Regex)

Standard spaCy NER does not recognize automotive-domain entities. Custom regex patterns:

| Entity Type | Regex Pattern | Example Match |
|-------------|--------------|---------------|
| BMW Signals | `BMW_[A-Za-z0-9_]+` | `BMW_t_HvsCellMinObd` |
| Temperature Thresholds | `(-?\d+\.?\d*)\s*°C` | `-40`, `215` |
| Qualifiers | `QUAL_[A-Z_]+` | `QUAL_INT_OK`, `QUAL_INT_DEBOUNCING` |
| Error Limit Values | `BMW_LIM_[A-Za-z0-9_]+` | `BMW_LIM_MAXERRTEMP_SC` |
| Logical Conditions (EN) | `\b(AND|OR|IF|WHEN|UNLESS)\b` | `AND`, `WHEN` |
| Logical Conditions (DE) | `\b(UND|ODER|WENN|FALLS)\b` | `UND`, `WENN` |

**Output per requirement:**
```json
{
  "signals":    ["BMW_t_HvsCellMinObd", "BMW_t_HvsCellCoreAct_rc"],
  "thresholds": ["-40", "215"],
  "qualifiers": ["QUAL_INT_OK", "QUAL_INT_DEBOUNCING"],
  "conditions": ["AND"],
  "error_vals": ["BMW_LIM_MAXERRTEMP_SC"]
}
```

### Step 4: TF-IDF Vectorization

**Configuration:**
```python
TfidfVectorizer(max_features=200, min_df=1, ngram_range=(1,1))
```

- `max_features=200`: Keep top 200 terms by TF-IDF score
- `min_df=1`: Include terms appearing in at least 1 document
- Computed separately for DE and EN, combined for clustering

**Why TF-IDF over simple count:** Downweights common BMS terms that appear in every
requirement (e.g., "ausgeben", "output") — highlights terms that distinguish requirements.

### Step 5: LDA Topic Modeling

**Configuration:**
```python
LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
```

Trained separately on German and English lemmatized text.

**Discovered topics (example from HVSCC-38653 dataset):**

| Topic | Top German Words | Top English Words | Label |
|-------|-----------------|-------------------|-------|
| 1 | temperatur, zell, kern, mittel | temperature, cell, core, average | Temperature Monitoring |
| 2 | qualifier, plausibel, fehler, ungültig | qualifier, plausib, invalid, fault | Fault & Plausibility |
| 3 | ausgabe, signal, wert, setzen | output, signal, value, set | Signal Output |
| 4 | obd, diagnose, pid, verfügbar | obd, diagnostic, pid, available | OBD Data |
| 5 | bereich, grenzwert, prüf, bedingung | range, limit, check, condition | Range & Conditions |

**Assignment:** Each requirement is assigned its dominant topic:
```python
df['topic_label_en'] = doc_topics.argmax(axis=1).map(TOPIC_LABELS_EN)
```

### Step 6: K-Means Clustering

**Configuration:**
```python
KMeans(n_clusters=4, random_state=42, n_init=10)
```

**Input:** Bilingual TF-IDF vectors (DE + EN concatenated), L2-normalized.
**Dimensionality reduction:** TruncatedSVD to 2D for visualization.

**Why 4 clusters:** Empirically found to produce well-separated groups for BMS OBD requirements.

**Cluster labels:**
1. Thermal / Thermisch
2. OBD Diagnostics / Diagnose
3. Fault Handling / Fehler
4. Plausibility / Plausibilität

### Step 7: Criticality Scoring

**Algorithm:**
```python
score = 0
for pattern in CRITICAL_PATTERNS_EN:   # 5 patterns
    if re.search(pattern, text_en): score += 1
for pattern in CRITICAL_PATTERNS_DE:   # 5 patterns
    if re.search(pattern, text_de): score += 1
is_critical = score >= 3
```

**Threshold = 3:** A requirement must match at least 3 distinct critical patterns
across both languages to be flagged. This reduces false positives from generic
requirements that use one critical word incidentally.

**Impact on TC generation:**
- `is_critical = True` → TC priority escalated to `Critical`
- HIL Fault Injection TC always generated for critical requirements
- End-to-End ECU TC priority set to `Critical`

### Step 8: Quality Analysis

**6 issue categories detected:**

| Issue | Pattern | Example | Language |
|-------|---------|---------|----------|
| vague_terms | appropriate, sufficient, normal | "within appropriate range" | EN |
| no_threshold | high, low, large, small | "high temperature detected" | EN |
| missing_unit | temperature, voltage without °/V | "temperature exceeds limit" | EN |
| vage_begriffe | geeignet, ausreichend | "geeigneter Wert" | DE |
| kein_schwellwert | hoch, niedrig, groß | "hoher Temperaturwert" | DE |
| fehlende_einheit | temperatur, spannung without unit | "Temperatur überschreitet" | DE |

---

## 3. LLM Phase — Step-by-Step

### Step 1: RAG Context Retrieval (ChromaDB)

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- Multilingual model — understands both German and English
- 384-dimensional dense vectors
- Query: concatenated EN + DE text of target requirement

**Retrieval:**
```python
results = collection.query(
    query_texts=[f"{text_en} {text_de}"],
    n_results=3
)
# Returns top-3 most semantically similar requirements
```

**Purpose:** Provides the LLM with examples of similar requirements and their structure,
improving test case relevance and consistency.

### Step 2: Prompt Construction

**System prompt (static):**
```
You are an expert automotive validation engineer specializing in
Battery Management Systems (BMS). Generate precise bilingual (DE+EN)
test cases following ISO 26262 principles.
Rules:
- Respond with valid JSON array only
- Cover nominal, boundary, fault scenarios
- Reference exact signal names, qualifiers, thresholds from requirement
- Both German and English fields mandatory
```

**User prompt (dynamic per requirement):**
```
Section: {section}
Criticality Score: {score}
Signals: {signals}
Thresholds: {thresholds} °C
Qualifiers: {qualifiers}

GERMAN REQUIREMENT (primary):
{text_de}

ENGLISH REQUIREMENT:
{text_en}

SIMILAR REQUIREMENTS FOR CONTEXT:
{rag_context}

Generate 3-5 test cases as JSON array.
```

**Few-shot example embedded in prompt:**
```json
[{
  "TC_ID": "TC_17_8_001",
  "Type_EN": "Nominal", "Type_DE": "Normalfall",
  "Objective_EN": "Verify BMW_t_HvsCellMinObd = input when OK",
  "Ziel_DE": "BMW_t_HvsCellMinObd = Eingangswert wenn i.O.",
  "Preconditions_EN": "Qualifier = QUAL_INT_OK; input in [-40°C, 215°C]",
  "Vorbedingungen_DE": "Qualifier = QUAL_INT_OK; Eingangswert in [-40°C, 215°C]",
  "Expected_Result_EN": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
  "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
  "Priority": "High",
  "Rationale": "Core nominal behavior — must pass for basic compliance"
}]
```

### Step 3: Ollama API Call

```python
response = ollama.chat(
    model="mistral",  # or llama3
    messages=[
        {"role": "system", "content": BMS_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt}
    ],
    options={"temperature": 0.2, "num_predict": 2000}
)
```

**Temperature = 0.2:** Low temperature for deterministic, precise output.
Higher values (>0.5) produce more creative but less reliable JSON.

### Step 4: Output Parsing

```python
raw = response["message"]["content"]
json_match = re.search(r'\[.*\]', raw, re.DOTALL)
if json_match:
    tcs = json.loads(json_match.group())
```

Robust parsing handles:
- Markdown code blocks (strips ```json ... ```)
- Extra text before/after the JSON array
- Nested objects with escaped characters

---

## 4. EN↔DE Consistency Check

### Heuristic Algorithm (implemented)
```python
# Signal overlap score
sigs_en = set(re.findall(r'BMW_\w+', text_en))
sigs_de = set(re.findall(r'BMW_\w+', text_de))
sig_overlap = len(sigs_en & sigs_de) / max(len(sigs_en | sigs_de), 1)

# Length ratio score
len_ratio = min(len_en, len_de) / max(len_en, len_de, 1)

# Composite score
consistency = (sig_overlap * 0.7 + len_ratio * 0.3) * 100
# Flag if < 60%
```

### Semantic Algorithm (planned — next step)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

emb_en = model.encode(text_en)
emb_de = model.encode(text_de)
similarity = cosine_similarity([emb_en], [emb_de])[0][0]
# Flag if similarity < 0.7
```

---

## 5. Pipeline Performance Metrics

| Step | Processing Time | Output |
|------|----------------|--------|
| Text cleaning | <0.1s / req | Clean string |
| spaCy DE lemmatize | ~0.5s / req | List of lemmas |
| spaCy EN lemmatize | ~0.3s / req | List of lemmas |
| Entity extraction | <0.1s / req | Dict with 5 entity types |
| TF-IDF vectorization | ~1s (all reqs) | Sparse matrix |
| LDA (5 topics, 20 iter) | ~3s (all reqs) | Topic distribution |
| K-Means clustering | ~1s (all reqs) | Cluster assignments |
| Criticality scoring | <0.1s / req | Integer score |
| Quality analysis | <0.1s / req | Issue flags |
| **Total NLP (50 reqs)** | **~15-30s** | Enriched DataFrame |
| ChromaDB indexing | ~10s (all reqs) | Vector store |
| LLM per requirement | 5-15s (GPU) / 30-60s (CPU) | 3-5 TCs |
| **Total LLM (50 reqs)** | **5-50 min** | Enhanced TCs |

---

## 6. Evaluation of Generated Test Cases

### Coverage Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Requirement Coverage | covered_reqs / total_reqs × 100 | ≥ 95% |
| Critical Coverage | covered_critical / total_critical × 100 | 100% |
| TC Approval Rate | approved_tcs / total_tcs × 100 | ≥ 80% |
| Avg TCs per Requirement | total_tcs / total_reqs | ≥ 3 |

### Quality Metrics

| Metric | Description | How Measured |
|--------|-------------|-------------|
| Bilingual completeness | Both DE and EN fields present | Assert non-null |
| TC ID uniqueness | No duplicate TC IDs | Assert nunique == len |
| Priority validity | Only Critical/High/Medium/Low | Assert subset |
| Entity traceability | Signal names from req appear in TC | String match |
| Threshold coverage | Each threshold has a Boundary TC | Check by section |

### Engineer Review Score

Engineers use the **Edit & Approve** dashboard to:
1. Review each TC — approve ✅ or edit ✏️
2. Approval rate tracked as quality indicator
3. Edited TCs stored separately — patterns fed back to LLM as few-shot examples

---

*Document: 02_NLP_AI_Pipeline.md — BMS GenAI Assistant v1.0*
