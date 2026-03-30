# BMS GenAI Assistant — Validation & Evaluation of Generated Test Cases

**Version:** 1.0

---

## 1. Evaluation Framework Overview

The generated test cases are evaluated across four dimensions:

```
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION DIMENSIONS                          │
│                                                             │
│  1. COVERAGE     2. QUALITY      3. ACCURACY   4. REVIEW   │
│  ─────────────  ─────────────   ───────────   ──────────  │
│  How many reqs  Are TCs well-   Do TCs match  Engineer    │
│  are tested?    formed?         the req?      approval %  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Coverage Metrics

### 2.1 Requirement Coverage
**Definition:** Percentage of requirements that have at least one generated test case.

```
Requirement Coverage = (Requirements with ≥1 TC) / Total Requirements × 100
```

**Measurement:**
```python
covered = df_unit["Section"].nunique()
total   = len(df_req)
req_coverage = round(covered / total * 100, 1)
```

**Target:** ≥ 95%  
**Result (HVSCC-38653 dataset):** 100% — all 50 requirements covered

---

### 2.2 Critical Requirement Coverage
**Definition:** Percentage of safety-critical requirements that have test cases.

```
Critical Coverage = (Critical Reqs with ≥1 TC) / Total Critical Reqs × 100
```

**Target:** 100% — no critical requirement left untested  
**Priority:** This metric must be 100% before release

---

### 2.3 Test Case Type Coverage
**Definition:** Percentage of requirements covered by each TC type.

| TC Type | Target Coverage |
|---------|----------------|
| Nominal | 100% (every req) |
| Boundary | 100% of reqs with thresholds |
| Fault | 100% of reqs with qualifiers/fault keywords |
| Out-of-Range | 100% of reqs with ≥1 threshold |
| ECU: End-to-End | 100% (every req) |
| ECU: OBD | 100% of OBD/PID/DTC reqs |
| ECU: CAN Bus | 100% of reqs with signals |
| ECU: HIL Fault | 100% of critical reqs |

---

### 2.4 Average TCs per Requirement

```
Avg TCs per Req = Total TCs / Total Requirements
```

**Target:** ≥ 4 (at least Nominal + Boundary + Fault + ECU E2E)  
**Result:** ~10 TCs per requirement (unit + ECU combined)

---

## 3. Quality Metrics

### 3.1 Structural Quality (Automated)

Checked by pytest (`tests/test_pipeline.py`):

| Check | Test | Expected |
|-------|------|---------|
| TC_ID uniqueness | `df["TC_ID"].nunique() == len(df)` | True |
| No null TC_IDs | `df["TC_ID"].notna().all()` | True |
| Valid priority values | `set(df["Priority"]) ⊆ {Critical,High,Medium,Low}` | True |
| Bilingual completeness | `df["Objective_EN"].notna().all()` | True |
| Bilingual completeness | `df["Ziel_DE"].notna().all()` | True |
| ECU columns present | All required columns exist | True |
| Entity traceability | Signal names match requirement | Spot check |

---

### 3.2 Bilingual Completeness Score

**Definition:** Percentage of TCs with all bilingual fields populated.

```python
bilingual_fields = [
    "Objective_EN", "Ziel_DE",
    "Preconditions_EN", "Vorbedingungen_DE",
    "Expected_Result_EN", "Erwartetes_Ergebnis_DE"
]
complete = df[bilingual_fields].notna().all(axis=1).mean() * 100
```

**Target:** 100% for rule-based TCs  
**Target:** ≥ 90% for LLM-generated TCs (DE fields sometimes weaker)

---

### 3.3 Requirement Traceability

**Definition:** Each TC must reference the correct requirement section and signal names.

```python
for _, row in df_unit.iterrows():
    req_section = row["Section"]
    tc_id       = row["TC_ID"]
    # TC_ID must start with TC_{section_with_underscores}
    assert tc_id.startswith(f"TC_{req_section.replace('.','_')}")
```

**Signal traceability (spot check):**
- Extract signals from requirement: `re.findall(r"BMW_\w+", req_text)`
- Verify at least 1 signal appears in TC `Inputs` or `Objective_EN`

---

## 4. Accuracy Evaluation

### 4.1 Nominal TC Accuracy
A nominal TC is accurate if:
- ✅ Precondition states `qualifier = QUAL_INT_OK`
- ✅ Input range matches requirement bounds (e.g., −40°C to 215°C)
- ✅ Expected result references the correct output signal

**Manual review sample:** 5 nominal TCs per topic category = 25 TCs reviewed

---

### 4.2 Boundary TC Accuracy
A boundary TC is accurate if:
- ✅ Threshold value matches requirement exactly (e.g., −40°C)
- ✅ Specifies upper OR lower boundary (not both in one TC)
- ✅ Expected result is not "blocked" (boundary is still valid)

---

### 4.3 Fault TC Accuracy
A fault TC is accurate if:
- ✅ Precondition specifies qualifier NOT OK (not QUAL_INT_OK, not QUAL_INT_DEBOUNCING)
- ✅ Expected result references the error/limit value (BMW_LIM_*) if present
- ✅ States "output not updated" or "Ausgangswert nicht aktualisiert"

---

### 4.4 LLM TC Accuracy Score

For LLM-generated TCs, a manual scoring rubric (1-5 per dimension):

| Dimension | 1 (Poor) | 3 (Acceptable) | 5 (Excellent) |
|-----------|----------|----------------|---------------|
| **Relevance** | TC unrelated to requirement | Partially related | Directly tests the stated behavior |
| **Correctness** | Wrong expected result | Partially correct | Matches requirement exactly |
| **Bilingual Quality** | DE/EN inconsistent | Minor inconsistencies | Both languages precise and aligned |
| **Signal Accuracy** | Wrong signals | Mix of correct/wrong | All signals from requirement |
| **Priority Appropriateness** | Wrong priority | Acceptable | Matches criticality score |

**Score calculation:**
```
TC_Score = (Relevance + Correctness + Bilingual + Signals + Priority) / 25 × 100
```

**Target:** Average TC score ≥ 75%

---

## 5. Engineer Review Process

### 5.1 Review Workflow

```
Generated TCs
    │
    ▼
1. Automated checks (pytest) — structural validation
    │
    ▼
2. Dashboard review (📝 Edit & Approve page)
   - Filter by type, priority, section
   - Review TC fields
   - Edit if needed
   - Approve ✅ or leave for revision
    │
    ▼
3. Approval rate recorded
    │
    ▼
4. Approved TCs → Export to ECU.TEST / Excel
```

### 5.2 Review Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Approval Rate | approved / total × 100 | ≥ 80% |
| Edit Rate | edited / total × 100 | ≤ 20% |
| Rejection Rate | not_approved / total × 100 | ≤ 5% |
| Review Time | Time to review 50 reqs of TCs | < 2 hours |

### 5.3 Review Focus Areas

Engineers should prioritize:
1. **Critical TCs** (Priority = Critical) — highest safety impact
2. **Fault TCs** — verify qualifier conditions are correct
3. **Boundary TCs** — verify exact threshold values match requirement
4. **ECU Integration TCs** — verify HIL setup and measurement tools are correct

---

## 6. Comparison: Manual vs. Generated TCs

### Time Savings

| Task | Manual (before) | Automated (after) |
|------|----------------|-------------------|
| Read requirement | 2 min | 0 (automated) |
| Write nominal TC | 5 min | 0 (generated) |
| Write boundary TC(s) | 5 min | 0 (generated) |
| Write fault TC | 5 min | 0 (generated) |
| Write ECU integration TCs | 15 min | 0 (generated) |
| **Total per requirement** | **~30 min** | **~2 min review** |
| **For 50 requirements** | **~25 hours** | **~2 hours review** |
| **Time reduction** | — | **~92%** |

### Quality Improvement

| Aspect | Manual | Generated |
|--------|--------|-----------|
| Coverage consistency | Depends on engineer | Always complete (4 unit + 6 ECU types) |
| Bilingual support | Rarely done | Always DE+EN |
| Criticality detection | Intuitive, subjective | Systematic, scored |
| Missed boundary cases | Common | Rare (always generated when thresholds found) |
| Documentation quality | Variable | Consistent format |

---

## 7. Automated Test Suite (`tests/test_pipeline.py`)

**30+ pytest tests covering:**

```
TestIngestion (4 tests)
├── test_load_excel
├── test_required_columns
├── test_only_anforderung
└── test_validation

TestNLPPipeline (8 tests)
├── test_tokens_generated
├── test_criticality_columns
├── test_some_critical
├── test_topics_assigned
├── test_clusters_assigned
├── test_bms_entities
├── test_ecu_level
└── test_quality_columns

TestTestCaseGenerator (10 tests)
├── test_unit_tcs_generated
├── test_ecu_tcs_generated
├── test_unit_tc_columns
├── test_ecu_tc_columns
├── test_tc_types_present
├── test_ecu_types_present
├── test_no_null_tc_ids
├── test_unique_tc_ids
├── test_bilingual_fields
└── test_priority_values

TestEntityExtraction (4 tests)
├── test_signal_extraction
├── test_threshold_extraction
├── test_qualifier_extraction
└── test_criticality_scorer
```

**Run:**
```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=. --cov-report=html
```

---

## 8. Continuous Evaluation in CI/CD

The pipeline stages automatically evaluate:

| Stage | What is Evaluated |
|-------|------------------|
| validate | Schema validity, missing fields |
| nlp | NLP output completeness |
| llm_generate | JSON parse success rate |
| ecu_tests | Required ECU TC columns present |
| coverage | Coverage % computed and reported |

The HTML coverage report (`outputs/coverage_report.html`) includes:
- Requirement coverage %
- Critical coverage %
- TC count by type
- List of uncovered requirements
- List of critical requirements needing attention

---

*Document: 04_Validation_Evaluation.md — BMS GenAI Assistant v1.0*
