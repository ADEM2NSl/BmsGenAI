# BMS GenAI Assistant — Prompt Engineering Methodology

**Version:** 1.0

---

## 1. Overview

Prompt engineering is the process of designing the inputs to the LLM to produce
reliable, structured, domain-specific test cases. This document describes the strategy
used to guide Mistral-7B / LLaMA-3 to generate high-quality BMS test cases.

---

## 2. Prompt Architecture

Every LLM call uses a **three-layer prompt structure:**

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: System Prompt (static)                    │
│  Role definition + output format constraints        │
├─────────────────────────────────────────────────────┤
│  LAYER 2: Few-Shot Example (static)                 │
│  One complete example of input → expected output    │
├─────────────────────────────────────────────────────┤
│  LAYER 3: Dynamic Context (per requirement)         │
│  ├── Extracted entities (signals, thresholds)       │
│  ├── Criticality score                              │
│  ├── German requirement text (primary)              │
│  ├── English requirement text (secondary)           │
│  └── RAG context (3 similar requirements)           │
└─────────────────────────────────────────────────────┘
```

---

## 3. System Prompt Design

### Principles Applied

**1. Role-first framing:**
```
"You are an expert automotive validation engineer specializing in
Battery Management Systems (BMS) for electric vehicles."
```
*Why:* Framing the model as a domain expert reduces generic responses
and increases use of automotive/safety terminology.

**2. Hard output constraints:**
```
"Always respond with valid JSON only — no markdown, no explanation
outside JSON, no preamble."
```
*Why:* Without this, Mistral often wraps output in markdown code blocks
or adds explanatory text that breaks JSON parsing.

**3. Safety-aware framing:**
```
"Follow ISO 26262 functional safety principles for critical requirements."
```
*Why:* BMS requirements have safety implications. This prompt element
increases likelihood of fault and boundary test cases for critical items.

**4. Explicit bilingual requirement:**
```
"Generate test cases in BOTH German (DE) and English (EN).
German is primary — Ziel_DE, Vorbedingungen_DE, Erwartetes_Ergebnis_DE.
English is secondary — Objective_EN, Preconditions_EN, Expected_Result_EN."
```
*Why:* Without explicit instruction, LLMs default to English only.
Specifying "German is primary" improves DE field quality.

### Full System Prompt
```
You are an expert automotive validation engineer specializing in
Battery Management Systems (BMS) for electric vehicles.
You generate precise, structured test cases for BMS ECU software validation.

RULES:
- Always respond with valid JSON only — no markdown, no explanation outside JSON
- Generate test cases in BOTH German (DE) and English (EN)
- Use BMS/automotive domain terminology precisely
- Cover nominal, boundary, fault, and out-of-range scenarios
- Reference specific signal names, qualifiers (QUAL_INT_OK), and temperature
  thresholds from the requirement
- Follow ISO 26262 functional safety principles for critical requirements

OUTPUT FORMAT (JSON array):
[{
  "TC_ID": "TC_X_X_001",
  "Type_EN": "Nominal|Boundary|Fault|Out-of-Range",
  "Type_DE": "Normalfall|Grenzwert|Fehlerfall|Bereichsüberschreitung",
  "Objective_EN": "...",
  "Ziel_DE": "...",
  "Preconditions_EN": "...",
  "Vorbedingungen_DE": "...",
  "Expected_Result_EN": "...",
  "Erwartetes_Ergebnis_DE": "...",
  "Priority": "Critical|High|Medium|Low",
  "Rationale": "Why this TC is needed"
}]
```

---

## 4. Few-Shot Example Design

### Why Few-Shot?
Zero-shot prompting produces varied output formats. One well-chosen example
is enough to lock in the expected structure, terminology, and bilingual pattern.

### Example Selection Criteria
The example was chosen to demonstrate:
1. A requirement with **threshold values** (−40°C to 215°C)
2. A requirement with **qualifiers** (QUAL_INT_OK, QUAL_INT_DEBOUNCING)
3. A requirement with **signal names** (BMW_t_HvsCellMinObd)
4. **Bilingual completeness** (all DE and EN fields populated)
5. A **fault scenario** (qualifier not OK case)

### The Few-Shot Example
```
EXAMPLE REQUIREMENT:
The BMU HL must output the minimum OBD cell core temperature
BMW_t_HvsCellMinObd (Unit: °C, Dim: 1x1) equal to
BMW_t_HvsCellCoreAct_rc.BMW_t_HvsCellCoreMinAct when the value
is plausible:
  - Qualifier = QUAL_INT_OK or QUAL_INT_DEBOUNCING
  - Value in range: −40°C to 215°C

EXAMPLE OUTPUT:
[
  {
    "TC_ID": "TC_17_8_001",
    "Type_EN": "Nominal",
    "Type_DE": "Normalfall",
    "Objective_EN": "Verify BMW_t_HvsCellMinObd equals input when qualifier OK and value in range",
    "Ziel_DE": "BMW_t_HvsCellMinObd entspricht Eingangswert wenn Qualifier i.O. und Wert im Bereich",
    "Preconditions_EN": "BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK; input in range [−40°C, 215°C]",
    "Vorbedingungen_DE": "BMW_t_HvsCellCoreAct_qi = QUAL_INT_OK; Eingangswert im Bereich [−40°C, 215°C]",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct",
    "Priority": "High",
    "Rationale": "Core nominal behavior — must pass for basic requirement compliance"
  },
  {
    "TC_ID": "TC_17_8_010",
    "Type_EN": "Boundary",
    "Type_DE": "Grenzwert",
    "Objective_EN": "Verify behavior at lower boundary −40°C",
    "Ziel_DE": "Verhalten an unterem Grenzwert −40°C prüfen",
    "Preconditions_EN": "Input = −40°C (lower boundary); qualifier = QUAL_INT_OK",
    "Vorbedingungen_DE": "Eingangswert = −40°C (unterer Grenzwert); Qualifier = QUAL_INT_OK",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = −40°C; no error set",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = −40°C; kein Fehler gesetzt",
    "Priority": "High",
    "Rationale": "Boundary value testing at minimum valid temperature"
  },
  {
    "TC_ID": "TC_17_8_099",
    "Type_EN": "Fault",
    "Type_DE": "Fehlerfall",
    "Objective_EN": "Verify behavior when qualifier is invalid (not OK)",
    "Ziel_DE": "Verhalten bei ungültigem Qualifier prüfen",
    "Preconditions_EN": "BMW_t_HvsCellCoreAct_qi ≠ QUAL_INT_OK and ≠ QUAL_INT_DEBOUNCING",
    "Vorbedingungen_DE": "BMW_t_HvsCellCoreAct_qi ≠ QUAL_INT_OK und ≠ QUAL_INT_DEBOUNCING",
    "Expected_Result_EN": "BMW_t_HvsCellMinObd = BMW_LIM_MAXERRTEMP_SC (error value); output not updated",
    "Erwartetes_Ergebnis_DE": "BMW_t_HvsCellMinObd = BMW_LIM_MAXERRTEMP_SC; Ausgangswert nicht aktualisiert",
    "Priority": "Critical",
    "Rationale": "Safety-critical: invalid qualifier must trigger safe default value"
  }
]
```

---

## 5. Dynamic Context Injection

### Entity Context Block
Extracted entities are injected before the requirement text:
```
Section: 17.8.68
Criticality Score: 5 (CRITICAL)
Signals: BMW_t_HvsCellMinObd, BMW_t_HvsCellCoreAct_rc
Thresholds: -40, 215 (°C)
Qualifiers: QUAL_INT_OK, QUAL_INT_DEBOUNCING
Error Values: BMW_LIM_MAXERRTEMP_SC
```

*Why:* Pre-extracted entities reduce hallucination of incorrect signal names.
The LLM can reference `{signals}` directly rather than parsing the requirement text.

### Criticality Weighting
For requirements with `criticality_score >= 3`:
```
⚠️ SAFETY-CRITICAL REQUIREMENT (Score: {score}/10)
Generate at least one FAULT test case and escalate priority to CRITICAL.
```

*Why:* Without explicit instruction, LLMs tend to generate more nominal TCs
even for fault-heavy requirements.

### RAG Context Format
```
SIMILAR REQUIREMENTS FOR CONTEXT:
---
[Requirement 1 text — most similar]
---
[Requirement 2 text]
---
[Requirement 3 text]
```

*Why:* Provides domain context from the same requirements document.
Helps the LLM understand patterns (e.g., all OBD requirements follow the same
qualifier structure) and generate consistent output.

---

## 6. LLM Parameter Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.2 | Low → deterministic, precise JSON. High values cause malformed JSON |
| `num_predict` | 2000 | Allows 3-5 complete TCs; lower values truncate output |
| `top_p` | default | Not customized — temperature control sufficient |
| `repeat_penalty` | default | Prevents repetition in long outputs |

### Temperature Trade-offs

| Temperature | Behavior | Use Case |
|-------------|----------|---------|
| 0.0 | Fully deterministic, same output every time | Regression testing |
| 0.1-0.2 | Mostly deterministic, minimal variation | **Production (recommended)** |
| 0.3-0.5 | Some creativity, still mostly correct | Exploratory generation |
| >0.5 | Creative but unreliable JSON structure | Not recommended |

---

## 7. Output Post-Processing

### JSON Extraction
```python
# Handle various LLM output formats:
# - Clean JSON array
# - Markdown fenced: ```json [...] ```
# - Text before/after the array

json_match = re.search(r'\[.*\]', raw, re.DOTALL)
if json_match:
    tcs = json.loads(json_match.group())
```

### Validation After Parsing
```python
required_fields = [
    "TC_ID", "Type_EN", "Type_DE",
    "Objective_EN", "Ziel_DE",
    "Expected_Result_EN", "Erwartetes_Ergebnis_DE",
    "Priority"
]
valid_tcs = [tc for tc in tcs if all(f in tc for f in required_fields)]
```

### Metadata Enrichment
After parsing, each LLM TC is tagged with:
```python
tc["Requirement_ID"] = requirement["Item ID"]
tc["Section"]        = requirement["Gliederungsnummer"]
tc["Category"]       = "LLM-Generated"
tc["Source"]         = f"Ollama/{model}"
tc["Generated_At"]   = datetime.now().isoformat()
```

---

## 8. Feedback Loop Design

### Current State
Engineer approvals and edits are stored in `SQLite` and session state.

### Planned Feedback Loop
```
Engineer edits TC in dashboard
    │
    ▼
Edited TC stored in ChromaDB with label "approved_example"
    │
    ▼
Next pipeline run: RAG retrieves approved examples
    │
    ▼
Approved examples added to few-shot section of prompt
    │
    ▼
LLM generates TCs more aligned with engineer preferences
```

This creates a **self-improving system** — the more engineers review and approve,
the better the LLM's outputs become over time.

---

## 9. Prompt Iteration History

| Version | Change | Result |
|---------|--------|--------|
| v0.1 | Simple zero-shot: "Generate test cases for this requirement" | Inconsistent format, English only |
| v0.2 | Added JSON schema in prompt | Format improved, still English only |
| v0.3 | Added bilingual instruction | German fields added, sometimes empty |
| v0.4 | Added few-shot example | Bilingual quality improved significantly |
| v0.5 | Added entity pre-extraction in prompt | Fewer hallucinated signal names |
| v0.6 | Added criticality weighting | More fault TCs for critical requirements |
| v1.0 | Added RAG context | More consistent terminology within req set |

---

## 10. Known Limitations & Mitigations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| JSON malformation | LLM sometimes adds trailing commas or text | `re.search` + `json.loads` with fallback |
| Signal hallucination | LLM invents signal names not in requirement | Entity pre-extraction + regex validation |
| German quality variation | DE fields sometimes weaker than EN | Temperature 0.2 + explicit DE-primary instruction |
| Long requirements | >400 chars may be truncated | Text chunking before injection |
| LLM unavailability | Ollama not running | Graceful fallback to rule-based TCs only |

---

*Document: 03_Prompt_Engineering.md — BMS GenAI Assistant v1.0*
