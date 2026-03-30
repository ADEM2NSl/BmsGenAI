"""
cicd/generate_report.py
Generates HTML coverage report from NLP + TC pipeline outputs
"""

import os
import glob
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from jinja2 import Template

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>BMS Test Coverage Report</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background:#050d1a; color:#cde3f0; margin:0; padding:24px; }
  h1 { color:#00d4ff; font-size:28px; } h2 { color:#00d4ff; font-size:18px; border-bottom:1px solid #0f2a4a; padding-bottom:8px; }
  .kpi-row { display:flex; gap:16px; margin:20px 0; flex-wrap:wrap; }
  .kpi { background:#0a1628; border:1px solid #0f2a4a; border-left:3px solid #00d4ff; border-radius:6px; padding:16px 24px; min-width:130px; }
  .kpi.red { border-left-color:#e74c3c; } .kpi.green { border-left-color:#39ff14; } .kpi.gold { border-left-color:#ffd700; }
  .kpi-val { font-size:32px; font-weight:700; color:#00d4ff; } .kpi.red .kpi-val { color:#e74c3c; } .kpi.green .kpi-val { color:#39ff14; } .kpi.gold .kpi-val { color:#ffd700; }
  .kpi-lbl { font-size:11px; color:#4a7a9b; text-transform:uppercase; letter-spacing:1px; }
  table { width:100%; border-collapse:collapse; margin:12px 0; font-size:12px; }
  th { background:#0a1628; color:#00d4ff; padding:8px 12px; text-align:left; border:1px solid #0f2a4a; }
  td { padding:7px 12px; border:1px solid #0f2a4a; } tr:nth-child(even) { background:#0a1628; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; }
  .badge-crit { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c44; }
  .badge-high { background:#f39c1222; color:#f39c12; border:1px solid #f39c1244; }
  .badge-ok   { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc7144; }
  .section { background:#0a1628; border:1px solid #0f2a4a; border-radius:6px; padding:20px; margin:20px 0; }
  footer { margin-top:40px; color:#4a7a9b; font-size:11px; text-align:center; border-top:1px solid #0f2a4a; padding-top:12px; }
</style>
</head>
<body>
<h1>🔋 BMS Test Coverage Report</h1>
<p style="color:#4a7a9b">Generated: {{ generated_at }} | Pipeline: {{ pipeline_id }}</p>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{{ total_reqs }}</div><div class="kpi-lbl">Requirements</div></div>
  <div class="kpi red"><div class="kpi-val">{{ critical_reqs }}</div><div class="kpi-lbl">Critical</div></div>
  <div class="kpi gold"><div class="kpi-val">{{ total_unit_tcs }}</div><div class="kpi-lbl">Unit TCs</div></div>
  <div class="kpi gold"><div class="kpi-val">{{ total_ecu_tcs }}</div><div class="kpi-lbl">ECU Integration TCs</div></div>
  <div class="kpi green"><div class="kpi-val">{{ coverage_pct }}%</div><div class="kpi-lbl">Req Coverage</div></div>
</div>

<div class="section">
  <h2>📊 Requirements Summary</h2>
  <table>
    <tr><th>Section</th><th>Status</th><th>Criticality</th><th>Topic (EN)</th><th>ECU Level</th><th>Unit TCs</th><th>ECU TCs</th></tr>
    {% for row in req_rows %}
    <tr>
      <td>{{ row.section }}</td>
      <td><span class="badge badge-ok">{{ row.status }}</span></td>
      <td>
        {% if row.is_critical %}<span class="badge badge-crit">CRITICAL ({{ row.score }})</span>
        {% else %}<span class="badge badge-ok">OK ({{ row.score }})</span>{% endif %}
      </td>
      <td>{{ row.topic_en }}</td>
      <td>{{ row.ecu_level }}</td>
      <td>{{ row.unit_tc_count }}</td>
      <td>{{ row.ecu_tc_count }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<div class="section">
  <h2>🔌 ECU Integration Test Types</h2>
  <table>
    <tr><th>Integration Type</th><th>Count</th><th>Priority Breakdown</th></tr>
    {% for row in ecu_type_rows %}
    <tr>
      <td>{{ row.type }}</td>
      <td>{{ row.count }}</td>
      <td>{{ row.priorities }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<div class="section">
  <h2>🔴 Critical Requirements Requiring Attention</h2>
  <table>
    <tr><th>Section</th><th>Score</th><th>ECU Level</th><th>Requirement (EN)</th></tr>
    {% for row in critical_rows %}
    <tr>
      <td>{{ row.section }}</td>
      <td><span class="badge badge-crit">{{ row.score }}</span></td>
      <td>{{ row.ecu_level }}</td>
      <td style="font-size:11px">{{ row.text_en[:150] }}...</td>
    </tr>
    {% endfor %}
  </table>
</div>

<footer>
  BMS GenAI Assistant — Automated Test Case Generation Pipeline<br/>
  Local Deployment · Open-Source LLM · spaCy DE+EN · Ollama/Mistral
</footer>
</body>
</html>"""


def generate_report():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Load NLP data
    nlp_files = glob.glob("outputs/nlp_*.json")
    unit_files = glob.glob("outputs/unit_tcs_*.xlsx")
    ecu_files  = glob.glob("outputs/ecu_tcs_*.xlsx")

    df_req   = pd.concat([pd.read_json(f) for f in nlp_files], ignore_index=True) if nlp_files else pd.DataFrame()
    df_unit  = pd.concat([pd.read_excel(f) for f in unit_files], ignore_index=True) if unit_files else pd.DataFrame()
    df_ecu   = pd.concat([pd.read_excel(f) for f in ecu_files], ignore_index=True) if ecu_files else pd.DataFrame()

    # Metrics
    total_reqs   = len(df_req)
    critical_reqs = int(df_req.get("is_critical", pd.Series()).sum()) if "is_critical" in df_req else 0
    total_unit   = len(df_unit)
    total_ecu    = len(df_ecu)
    covered_reqs = df_unit["Section"].nunique() if "Section" in df_unit.columns else 0
    coverage_pct = round(covered_reqs / total_reqs * 100) if total_reqs > 0 else 0

    # Req rows
    req_rows = []
    if not df_req.empty and "Gliederungsnummer" in df_req.columns:
        for _, row in df_req.iterrows():
            sec = str(row.get("Gliederungsnummer", ""))
            unit_count = len(df_unit[df_unit["Section"] == sec]) if not df_unit.empty else 0
            ecu_count  = len(df_ecu[df_ecu["Section"] == sec]) if not df_ecu.empty else 0
            req_rows.append({
                "section":      sec,
                "status":       row.get("Status", ""),
                "is_critical":  row.get("is_critical", False),
                "score":        row.get("criticality_score", 0),
                "topic_en":     row.get("topic_label_en", ""),
                "ecu_level":    row.get("ecu_level", ""),
                "unit_tc_count": unit_count,
                "ecu_tc_count":  ecu_count,
            })

    # ECU type breakdown
    ecu_type_rows = []
    if not df_ecu.empty and "Integration_Type_EN" in df_ecu.columns:
        for itype, grp in df_ecu.groupby("Integration_Type_EN"):
            prio_str = ", ".join(f"{k}:{v}" for k, v in grp["Priority"].value_counts().items())
            ecu_type_rows.append({"type": itype, "count": len(grp), "priorities": prio_str})

    # Critical rows
    critical_rows = []
    if not df_req.empty and "is_critical" in df_req.columns:
        crit_df = df_req[df_req["is_critical"] == True]
        for _, row in crit_df.iterrows():
            critical_rows.append({
                "section":   str(row.get("Gliederungsnummer", "")),
                "score":     row.get("criticality_score", 0),
                "ecu_level": row.get("ecu_level", ""),
                "text_en":   str(row.get("text_en", ""))[:200],
            })

    # Render
    tmpl = Template(TEMPLATE)
    html = tmpl.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pipeline_id=os.getenv("CI_PIPELINE_ID", "local"),
        total_reqs=total_reqs,
        critical_reqs=critical_reqs,
        total_unit_tcs=total_unit,
        total_ecu_tcs=total_ecu,
        coverage_pct=coverage_pct,
        req_rows=req_rows,
        ecu_type_rows=ecu_type_rows,
        critical_rows=critical_rows,
    )

    out_path = out_dir / "coverage_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Coverage report: {out_path}")
    print(f"   Requirements: {total_reqs} | Critical: {critical_reqs} | Coverage: {coverage_pct}%")


if __name__ == "__main__":
    generate_report()
