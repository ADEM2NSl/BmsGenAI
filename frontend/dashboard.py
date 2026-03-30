"""
frontend/dashboard.py
Streamlit Web Dashboard — BMS GenAI Assistant v1.1
Complete: TC editing, evaluation metrics, prompt customization,
manual TC builder, EN↔DE consistency check, feedback loop.
Run with: streamlit run frontend/dashboard.py
"""

import sys, os, json, uuid, requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="BMS GenAI Assistant", page_icon="🔋",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Exo+2:wght@400;500&display=swap');
  h1,h2,h3{font-family:'Rajdhani',sans-serif!important;}
  .kpi-card{background:#0a1628;border:1px solid #0f2a4a;border-left:3px solid #00d4ff;
    border-radius:6px;padding:16px;text-align:center;margin-bottom:8px;}
  .kpi-value{font-family:'Rajdhani',sans-serif;font-size:36px;font-weight:700;color:#00d4ff;}
  .kpi-label{font-size:11px;color:#4a7a9b;letter-spacing:1px;text-transform:uppercase;}
  .kpi-card.red{border-left-color:#e74c3c;}.kpi-card.red .kpi-value{color:#e74c3c;}
  .kpi-card.green{border-left-color:#39ff14;}.kpi-card.green .kpi-value{color:#39ff14;}
  .kpi-card.gold{border-left-color:#ffd700;}.kpi-card.gold .kpi-value{color:#ffd700;}
  .kpi-card.purple{border-left-color:#c850c0;}.kpi-card.purple .kpi-value{color:#c850c0;}
  .info-box{background:#0a1628;border:1px solid #0f2a4a;border-radius:6px;padding:14px 16px;font-size:13px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "token":None,"run_id":None,"tc_run_id":None,"username":"engineer",
    "chat_history":[],"df_reqs":None,"df_unit":None,"df_ecu":None,
    "edited_tcs":{},"manual_tcs":[],"approved_tcs":set(),
    "llm_temperature":0.2,"llm_focus":"All",
}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

# ── API helpers ───────────────────────────────────────────────────────────────
def hdrs(): return {"Authorization":f"Bearer {st.session_state.token}"}

def api_get(ep):
    try:
        r=requests.get(f"{API_URL}{ep}",headers=hdrs(),timeout=30)
        r.raise_for_status(); return r.json()
    except Exception as e: st.error(f"API: {e}"); return None

def api_post(ep,data=None,files=None,json_data=None):
    try:
        kw={"headers":{"Authorization":f"Bearer {st.session_state.token}"}if files else hdrs(),
            "timeout":180}
        if files: kw["files"]=files
        elif json_data: kw["json"]=json_data
        elif data: kw["data"]=data
        r=requests.post(f"{API_URL}{ep}",**kw)
        r.raise_for_status(); return r.json()
    except Exception as e: st.error(f"API: {e}"); return None

def kpi(val,lbl,color=""):
    st.markdown(f'<div class="kpi-card {color}"><div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{lbl}</div></div>',unsafe_allow_html=True)

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def login_page():
    _,c,_=st.columns([1,2,1])
    with c:
        st.markdown("# 🔋 BMS GenAI Assistant")
        st.markdown("##### Automated Test Case Generation for BMS ECU")
        st.divider()
        with st.form("login"):
            u=st.text_input("Username",value="engineer")
            p=st.text_input("Password",type="password",value="bms2024")
            ok=st.form_submit_button("🔐 Login",use_container_width=True)
        if ok:
            try:
                r=requests.post(f"{API_URL}/auth/token",data={"username":u,"password":p},timeout=10)
                if r.status_code==200:
                    d=r.json()
                    st.session_state.token=d["access_token"]
                    st.session_state.username=d["username"]
                    st.success("✅ Logged in!"); st.rerun()
                else: st.error("❌ Invalid credentials")
            except: st.warning("⚠️ Cannot reach API.\n\n`uvicorn backend.api:app --port 8000`")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### 🔋 BMS GenAI")
        st.caption(f"👤 **{st.session_state.username}**")
        st.divider()
        page=st.radio("Navigation",[
            "📊 Dashboard","📥 Upload Requirements","🧠 NLP Analysis",
            "🧪 Unit Test Cases","🔌 ECU Integration","📝 Edit & Approve",
            "➕ Manual TC Builder","📈 Evaluation Metrics","💬 Chat (RAG)",
            "⚙️ LLM Settings","📤 Export",
        ])
        st.divider()
        try:
            h=requests.get(f"{API_URL}/health",timeout=3).json()
            st.success("🟢 API Online")
            st.caption(f"Ollama: {'🟢' if h.get('ollama_available') else '🔴'}")
        except: st.error("🔴 API Offline")
        if st.button("🚪 Logout",use_container_width=True):
            for k,v in DEFAULTS.items(): st.session_state[k]=v
            st.rerun()
    return page

# ── PAGE: DASHBOARD ───────────────────────────────────────────────────────────
def page_dashboard():
    st.title("📊 Dashboard")
    if not st.session_state.run_id:
        st.info("👆 Upload a requirements file first."); return
    stats=api_get(f"/requirements/{st.session_state.run_id}/stats") or {}
    tc=api_get(f"/testcases/{st.session_state.tc_run_id}") if st.session_state.tc_run_id else None
    total_tc=(tc["unit_count"]+tc["ecu_count"]+tc.get("llm_count",0)) if tc else 0
    approved=len(st.session_state.approved_tcs)
    cov=round(approved/(total_tc+len(st.session_state.manual_tcs))*100) if (total_tc+len(st.session_state.manual_tcs))>0 else 0
    c1,c2,c3,c4,c5,c6=st.columns(6)
    with c1: kpi(stats.get("total",0),"Requirements")
    with c2: kpi(stats.get("critical",0),"Critical","red")
    with c3: kpi(total_tc,"Generated TCs","gold")
    with c4: kpi(len(st.session_state.manual_tcs),"Manual TCs","purple")
    with c5: kpi(approved,"Approved","green")
    with c6: kpi(f"{cov}%","Approved %","green")
    st.divider()
    if st.session_state.df_reqs is not None:
        df=st.session_state.df_reqs
        c1,c2=st.columns(2)
        with c1:
            if "topic_label_en" in df.columns:
                fig=px.pie(df,names="topic_label_en",title="Requirements by Topic (EN)",
                           color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=300)
                st.plotly_chart(fig,use_container_width=True)
        with c2:
            if "criticality_score" in df.columns:
                fig=px.histogram(df,x="criticality_score",title="Criticality Score Distribution",
                                 color_discrete_sequence=["#e74c3c"])
                fig.add_vline(x=3,line_dash="dash",line_color="#ffd700",annotation_text="Critical ≥3")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0a1628",font_color="#cde3f0",height=300)
                st.plotly_chart(fig,use_container_width=True)

# ── PAGE: UPLOAD ──────────────────────────────────────────────────────────────
def page_upload():
    st.title("📥 Upload Requirements")
    f=st.file_uploader("Choose Excel file (.xlsx)",type=["xlsx","xls"])
    c1,c2=st.columns(2)
    with c1: use_llm=st.toggle("🤖 Enable LLM (Ollama/Mistral)",value=False)
    with c2: st.caption("Requires `ollama serve` + `ollama pull mistral`")
    if f and st.button("🚀 Run Pipeline",type="primary",use_container_width=True):
        with st.spinner("Running NLP pipeline..."):
            res=api_post("/requirements/upload",files={
                "file":(f.name,f.getvalue(),"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        if res:
            st.session_state.run_id=res["run_id"]
            st.success(f"✅ Done! Run ID: `{res['run_id']}`")
            c1,c2,c3=st.columns(3)
            with c1: st.metric("Requirements",res["requirements_count"])
            with c2: st.metric("Critical",res["critical_count"])
            with c3: st.metric("Valid","✅" if res["validation"]["valid"] else "⚠️")
            for i in res["validation"].get("issues",[]): st.warning(f"⚠️ {i}")
            data=api_get(f"/requirements/{res['run_id']}")
            if data: st.session_state.df_reqs=pd.DataFrame(data["requirements"])
            with st.spinner("Generating test cases..."):
                tc=api_post(f"/testcases/generate/{res['run_id']}?use_llm={str(use_llm).lower()}")
            if tc:
                st.session_state.tc_run_id=tc["tc_run_id"]
                td=api_get(f"/testcases/{tc['tc_run_id']}")
                if td:
                    st.session_state.df_unit=pd.DataFrame(td.get("unit_tcs",[]))
                    st.session_state.df_ecu =pd.DataFrame(td.get("ecu_tcs",[]))
                st.info(f"🧪 {tc['total']} TCs ({tc['unit_tcs']} unit · {tc['ecu_tcs']} ECU · {tc.get('llm_tcs',0)} LLM)")

# ── PAGE: NLP ─────────────────────────────────────────────────────────────────
def page_nlp():
    st.title("🧠 NLP Analysis")
    if st.session_state.df_reqs is None: st.info("Upload requirements first."); return
    df=st.session_state.df_reqs
    t1,t2,t3,t4,t5=st.tabs(["📋 Requirements","🗂️ Topics","🔴 Criticality","⚠️ Quality","🔍 EN↔DE Check"])
    with t1:
        cols=["Gliederungsnummer","Status","len_words_en","criticality_score","is_critical","topic_label_en","ecu_level","text_en"]
        st.dataframe(df[[c for c in cols if c in df.columns]],use_container_width=True,height=420)
    with t2:
        c1,c2=st.columns(2)
        for col,ax,title,cmap in[("topic_label_en",c1,"🇬🇧 Topics — EN",px.colors.qualitative.Set2),
                                  ("topic_label_de",c2,"🇩🇪 Themen — DE",px.colors.qualitative.Pastel)]:
            if col in df.columns:
                with ax:
                    fig=px.pie(df,names=col,title=title,color_discrete_sequence=cmap)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=320)
                    st.plotly_chart(fig,use_container_width=True)
    with t3:
        if "criticality_score" in df.columns:
            fig=px.scatter(df,x="len_words_en",y="criticality_score",color="is_critical",
                           size="n_signals",hover_data=["Gliederungsnummer"],
                           color_discrete_map={True:"#e74c3c",False:"#2ecc71"},
                           title="Complexity vs Criticality")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=400)
            st.plotly_chart(fig,use_container_width=True)
            crit=df[df.get("is_critical",pd.Series(dtype=bool))==True]
            if len(crit): st.dataframe(crit[["Gliederungsnummer","criticality_score","text_en"]],use_container_width=True)
    with t4:
        ic=[c for c in df.columns if c.startswith("issue_")]
        if ic:
            s=df[ic].sum().sort_values(ascending=False)
            lbl=[c.replace("issue_en_","🇬🇧 ").replace("issue_de_","🇩🇪 ").replace("_"," ") for c in s.index]
            fig=px.bar(x=lbl,y=s.values,title="Quality Issues",color_discrete_sequence=["#f39c12"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=350)
            st.plotly_chart(fig,use_container_width=True)
    with t5:
        st.markdown("### 🔍 EN↔DE Consistency Check")
        import re
        rows=[]
        for _,row in df.iterrows():
            sigs_en=set(re.findall(r"BMW_\w+",str(row.get("text_en",""))))
            sigs_de=set(re.findall(r"BMW_\w+",str(row.get("text_de",""))))
            sig_ov=len(sigs_en&sigs_de)/max(len(sigs_en|sigs_de),1)
            en_w=len(str(row.get("text_en","")).split())
            de_w=len(str(row.get("text_de","")).split())
            lr=min(en_w,de_w)/max(en_w,de_w,1)
            score=round((sig_ov*0.7+lr*0.3)*100)
            rows.append({"Section":row["Gliederungsnummer"],"Signal Overlap%":round(sig_ov*100),
                         "Length Ratio%":round(lr*100),"Score":score,"Flag":"⚠️ CHECK" if score<60 else "✅ OK"})
        df_c=pd.DataFrame(rows).sort_values("Score")
        st.dataframe(df_c,use_container_width=True,height=380)
        flagged=(df_c["Flag"]=="⚠️ CHECK").sum()
        if flagged: st.warning(f"⚠️ {flagged} requirements may have EN↔DE inconsistencies (score < 60%)")
        st.caption("💡 For semantic similarity: use `paraphrase-multilingual-MiniLM-L12-v2` from sentence-transformers")

# ── PAGE: UNIT TCs ────────────────────────────────────────────────────────────
def page_unit_tcs():
    st.title("🧪 Unit Test Cases")
    if st.session_state.df_unit is None or len(st.session_state.df_unit)==0:
        st.info("Upload requirements first."); return
    df=st.session_state.df_unit.copy()
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi(len(df),"Total Unit TCs")
    with c2: kpi(int((df["Type_EN"]=="Nominal").sum()),"Nominal")
    with c3: kpi(int((df["Type_EN"]=="Boundary").sum()),"Boundary","gold")
    with c4: kpi(int((df["Priority"]=="Critical").sum()),"Critical","red")
    st.divider()
    c1,c2=st.columns(2)
    with c1: st_type=st.selectbox("Type",["All"]+list(df["Type_EN"].unique()))
    with c2: st_prio=st.selectbox("Priority",["All"]+list(df["Priority"].unique()))
    filt=df.copy()
    if st_type!="All": filt=filt[filt["Type_EN"]==st_type]
    if st_prio!="All": filt=filt[filt["Priority"]==st_prio]
    cols=["TC_ID","Section","Type_EN","Type_DE","Priority","Objective_EN","Ziel_DE",
          "Preconditions_EN","Expected_Result_EN","Thresholds_C","Inputs","Criticality_Score"]
    st.dataframe(filt[[c for c in cols if c in filt.columns]],use_container_width=True,height=440)
    st.divider()
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("✅ Approve ALL filtered",use_container_width=True):
            st.session_state.approved_tcs|=set(filt["TC_ID"].tolist())
            st.success(f"✅ {len(filt)} TCs approved!")
    with c2:
        if st.button("✅ Approve Critical only",use_container_width=True):
            ids=set(filt[filt["Priority"]=="Critical"]["TC_ID"].tolist())
            st.session_state.approved_tcs|=ids; st.success(f"✅ {len(ids)} approved!")
    with c3: st.caption(f"Approved: **{len(st.session_state.approved_tcs)}**")

# ── PAGE: ECU ─────────────────────────────────────────────────────────────────
def page_ecu():
    st.title("🔌 ECU Integration Test Cases")
    if st.session_state.df_ecu is None or len(st.session_state.df_ecu)==0:
        st.info("Upload requirements first."); return
    df=st.session_state.df_ecu.copy()
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi(len(df),"Total ECU TCs")
    with c2: kpi(int((df.get("Integration_Type_EN",pd.Series())=="HIL Fault Injection").sum()),"HIL Fault","red")
    with c3: kpi(int((df.get("Integration_Type_EN",pd.Series())=="OBD / Diagnostic Interface").sum()),"OBD/UDS","gold")
    with c4: kpi(int((df.get("Priority",pd.Series())=="Critical").sum()),"Critical","red")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        if "Integration_Type_EN" in df.columns:
            fig=px.pie(df,names="Integration_Type_EN",title="By Integration Type",
                       color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=300)
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        if "ECU_Level" in df.columns:
            fig=px.bar(df["ECU_Level"].value_counts().reset_index(),x="ECU_Level",y="count",
                       title="By ECU Level",color_discrete_sequence=["#39ff14"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=300)
            st.plotly_chart(fig,use_container_width=True)
    sel=st.selectbox("Filter",["All"]+list(df.get("Integration_Type_EN",pd.Series()).unique()))
    filt=df if sel=="All" else df[df["Integration_Type_EN"]==sel]
    cols=["TC_ID","Section","Integration_Type_EN","ECU_Level","Priority","Objective_EN",
          "Ziel_DE","Test_Environment","Expected_Result_EN","Measurement_Tool"]
    st.dataframe(filt[[c for c in cols if c in filt.columns]],use_container_width=True,height=400)

# ── PAGE: EDIT & APPROVE ──────────────────────────────────────────────────────
def page_edit():
    st.title("📝 Edit & Approve Test Cases")
    if st.session_state.df_unit is None: st.info("Upload requirements first."); return
    parts=[st.session_state.df_unit.assign(Source="Unit")]
    if st.session_state.df_ecu is not None and len(st.session_state.df_ecu):
        parts.append(st.session_state.df_ecu.assign(Source="ECU"))
    if st.session_state.manual_tcs:
        parts.append(pd.DataFrame(st.session_state.manual_tcs).assign(Source="Manual"))
    df_all=pd.concat(parts,ignore_index=True)
    if "TC_ID" not in df_all.columns: st.warning("No TCs."); return
    sel_id=st.selectbox("Select TC to edit",df_all["TC_ID"].tolist())
    row=df_all[df_all["TC_ID"]==sel_id].iloc[0].to_dict()
    st.divider()
    st.markdown(f"**Editing:** `{sel_id}` | Source: `{row.get('Source','')}` | Priority: `{row.get('Priority','')}`")
    c1,c2=st.columns(2)
    with c1:
        oe=st.text_area("Objective (EN)",value=str(row.get("Objective_EN","")),height=80)
        pe=st.text_area("Preconditions (EN)",value=str(row.get("Preconditions_EN","")),height=80)
        ee=st.text_area("Expected Result (EN)",value=str(row.get("Expected_Result_EN","")),height=80)
    with c2:
        od=st.text_area("Ziel (DE)",value=str(row.get("Ziel_DE","")),height=80)
        pd_=st.text_area("Vorbedingungen (DE)",value=str(row.get("Vorbedingungen_DE","")),height=80)
        ed=st.text_area("Erwartetes Ergebnis (DE)",value=str(row.get("Erwartetes_Ergebnis_DE","")),height=80)
    np_=st.select_slider("Priority",["Low","Medium","High","Critical"],value=str(row.get("Priority","High")))
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("💾 Save Edits",type="primary",use_container_width=True):
            st.session_state.edited_tcs[sel_id]={
                "TC_ID":sel_id,"Objective_EN":oe,"Ziel_DE":od,
                "Preconditions_EN":pe,"Vorbedingungen_DE":pd_,
                "Expected_Result_EN":ee,"Erwartetes_Ergebnis_DE":ed,
                "Priority":np_,"Edited_By":st.session_state.username,
                "Edited_At":datetime.now().isoformat()}
            st.success(f"✅ Saved edits for {sel_id}")
    with c2:
        if st.button("✅ Approve",use_container_width=True):
            st.session_state.approved_tcs.add(sel_id); st.success("✅ Approved!")
    with c3:
        if sel_id in st.session_state.approved_tcs:
            if st.button("↩️ Revoke",use_container_width=True):
                st.session_state.approved_tcs.discard(sel_id); st.warning("Revoked.")
    if st.session_state.edited_tcs:
        st.divider()
        st.markdown(f"### ✏️ {len(st.session_state.edited_tcs)} Edited TCs")
        st.dataframe(pd.DataFrame(st.session_state.edited_tcs.values()),use_container_width=True)
    st.divider()
    st.markdown(f"### ✅ Approved: {len(st.session_state.approved_tcs)} / {len(df_all)}")
    if st.session_state.approved_tcs:
        st.dataframe(df_all[df_all["TC_ID"].isin(st.session_state.approved_tcs)][
            ["TC_ID","Source","Priority","Objective_EN"]],use_container_width=True,height=250)

# ── PAGE: MANUAL TC BUILDER ───────────────────────────────────────────────────
def page_manual():
    st.title("➕ Manual Test Case Builder")
    st.caption("Create custom test cases not derived from requirements.")
    with st.form("manual_tc"):
        c1,c2=st.columns(2)
        with c1:
            tc_id=st.text_input("TC ID",value=f"MAN_{uuid.uuid4().hex[:6].upper()}")
            section=st.text_input("Section","Custom")
            tc_type=st.selectbox("Type",["Nominal","Boundary","Fault","Out-of-Range","Regression","Exploratory"])
            prio=st.selectbox("Priority",["Critical","High","Medium","Low"])
        with c2:
            env=st.selectbox("Environment",["SIL","HIL","MIL","Desk Check"])
            inputs=st.text_input("Input Signals","N/A")
            thr=st.text_input("Thresholds (°C)","N/A")
        c1,c2=st.columns(2)
        with c1:
            oe=st.text_area("Objective (EN)",height=80)
            pe=st.text_area("Preconditions (EN)",height=80)
            ee=st.text_area("Expected Result (EN)",height=80)
        with c2:
            od=st.text_area("Ziel (DE)",height=80)
            pd_=st.text_area("Vorbedingungen (DE)",height=80)
            ed=st.text_area("Erwartetes Ergebnis (DE)",height=80)
        sub=st.form_submit_button("➕ Add TC",type="primary",use_container_width=True)
    if sub:
        st.session_state.manual_tcs.append({
            "TC_ID":tc_id,"Section":section,"Category":"Manual","Type_EN":tc_type,"Type_DE":tc_type,
            "Priority":prio,"Test_Environment":env,"Inputs":inputs,"Thresholds_C":thr,
            "Objective_EN":oe,"Ziel_DE":od,"Preconditions_EN":pe,"Vorbedingungen_DE":pd_,
            "Expected_Result_EN":ee,"Erwartetes_Ergebnis_DE":ed,
            "Created_By":st.session_state.username,"Created_At":datetime.now().isoformat(),"Criticality_Score":0})
        st.success(f"✅ Added `{tc_id}`")
    if st.session_state.manual_tcs:
        st.divider()
        st.markdown(f"### 📋 {len(st.session_state.manual_tcs)} Manual TCs")
        st.dataframe(pd.DataFrame(st.session_state.manual_tcs)[
            ["TC_ID","Type_EN","Priority","Objective_EN","Test_Environment"]],use_container_width=True)
        if st.button("🗑️ Clear all"): st.session_state.manual_tcs=[]; st.rerun()

# ── PAGE: EVALUATION METRICS ──────────────────────────────────────────────────
def page_metrics():
    st.title("📈 Evaluation Metrics")
    if st.session_state.df_reqs is None or st.session_state.df_unit is None:
        st.info("Upload requirements first."); return
    df_req=st.session_state.df_reqs
    df_u=st.session_state.df_unit
    df_e=st.session_state.df_ecu if st.session_state.df_ecu is not None else pd.DataFrame()
    df_m=pd.DataFrame(st.session_state.manual_tcs)
    total_req=len(df_req); total_u=len(df_u); total_e=len(df_e)
    total_m=len(df_m); total_tc=total_u+total_e+total_m
    approved=len(st.session_state.approved_tcs); edited=len(st.session_state.edited_tcs)
    crit_reqs=int(df_req.get("is_critical",pd.Series(dtype=bool)).sum()) if "is_critical" in df_req.columns else 0
    cov_reqs=df_u["Section"].nunique() if "Section" in df_u.columns else 0
    req_cov=round(cov_reqs/total_req*100) if total_req else 0
    tc_app_pct=round(approved/total_tc*100) if total_tc else 0
    avg_per_req=round(total_tc/total_req,1) if total_req else 0
    if "Gliederungsnummer" in df_req.columns and "is_critical" in df_req.columns and "Section" in df_u.columns:
        cs=set(df_req[df_req["is_critical"]==True]["Gliederungsnummer"].astype(str))
        cc=set(df_u["Section"].astype(str))&cs
        crit_cov=round(len(cc)/len(cs)*100) if cs else 100
    else: crit_cov=0
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: kpi(f"{req_cov}%","Req Coverage","green")
    with c2: kpi(f"{crit_cov}%","Critical Cov.","red" if crit_cov<80 else "green")
    with c3: kpi(f"{tc_app_pct}%","TCs Approved","green")
    with c4: kpi(avg_per_req,"Avg TCs/Req")
    with c5: kpi(edited,"TCs Edited","gold")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(x=["Unit","ECU","Manual"],y=[total_u,total_e,total_m],title="TCs by Category",
                   color=["Unit","ECU","Manual"],
                   color_discrete_map={"Unit":"#3498db","ECU":"#39ff14","Manual":"#c850c0"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=280,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        if "Type_EN" in df_u.columns:
            tc=df_u["Type_EN"].value_counts()
            fig=px.pie(values=tc.values,names=tc.index,title="Unit TC Types",
                       color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#cde3f0",height=280)
            st.plotly_chart(fig,use_container_width=True)
    st.divider()
    st.markdown("### 📋 Per-Requirement Coverage")
    if "Gliederungsnummer" in df_req.columns and "Section" in df_u.columns:
        rows=[]
        for _,r in df_req.iterrows():
            sec=str(r["Gliederungsnummer"])
            un=len(df_u[df_u["Section"]==sec])
            en=len(df_e[df_e["Section"]==sec]) if not df_e.empty and "Section" in df_e.columns else 0
            ic=bool(r.get("is_critical",False))
            ap=sum(1 for t in st.session_state.approved_tcs
                   if t.startswith(f"TC_{sec.replace('.','_')}") or t.startswith(f"ECU_TC_{sec.replace('.','_')}"))
            rows.append({"Section":sec,"Status":r.get("Status",""),"Critical":"🔴" if ic else "🟢",
                         "Unit TCs":un,"ECU TCs":en,"Total":un+en,"Approved":ap,
                         "Coverage":"✅" if(un+en)>0 else "❌"})
        df_cov=pd.DataFrame(rows)
        st.dataframe(df_cov,use_container_width=True,height=380)
        unc=(df_cov["Coverage"]=="❌").sum()
        if unc: st.warning(f"⚠️ {unc} requirements have no test cases!")
    st.divider()
    st.markdown(f"""
    <div class="info-box">
    <b>Summary</b><br/>
    Requirements: {total_req} total · {crit_reqs} critical · {cov_reqs} covered ({req_cov}%)<br/>
    Test Cases: {total_u} unit · {total_e} ECU · {total_m} manual = <b>{total_tc} total</b><br/>
    Approved: <b>{approved} ({tc_app_pct}%)</b> · Edited: <b>{edited}</b> · Avg per req: <b>{avg_per_req}</b>
    </div>""",unsafe_allow_html=True)

# ── PAGE: CHAT ────────────────────────────────────────────────────────────────
def page_chat():
    st.title("💬 Chat (RAG + Mistral)")
    st.caption("Ask in English or German. Powered by local Mistral + ChromaDB RAG.")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    q=st.chat_input("Ask about BMS requirements (EN or DE)...")
    if q:
        st.session_state.chat_history.append({"role":"user","content":q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res=api_post("/chat",json_data={"question":q})
            ans=res.get("answer","No answer.") if res else "❌ LLM not available — run `ollama serve`"
            st.markdown(ans)
            st.session_state.chat_history.append({"role":"assistant","content":ans})
    c1,c2=st.columns(2)
    with c1:
        if st.session_state.chat_history and st.button("🗑️ Clear"):
            st.session_state.chat_history=[]; st.rerun()
    with c2: st.caption("💡 Try: *'Which requirements are critical?'* or *'Welche Anforderungen haben Grenzwerte?'*")

# ── PAGE: LLM SETTINGS ────────────────────────────────────────────────────────
def page_llm():
    st.title("⚙️ LLM Settings & Prompt Customization")

    from llm.generator import SUPPORTED_MODELS, DEFAULT_MODEL

    st.markdown("### 🤖 Model Selection")
    st.caption("Cloud models give much better results than local models for BMS test case generation.")

    # Model selector with descriptions
    model_options = list(SUPPORTED_MODELS.keys())
    model_labels  = [
        f"{m}  —  {SUPPORTED_MODELS[m]['description']}"
        for m in model_options
    ]
    default_idx = model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 0
    sel_label = st.selectbox("Select Model", model_labels, index=default_idx)
    sel_model = model_options[model_labels.index(sel_label)]
    cfg       = SUPPORTED_MODELS[sel_model]

    # Show model card
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Default Temperature", cfg["temperature"])
        st.caption("Lower = more precise JSON output")
    with c2:
        st.metric("Max Tokens", cfg["num_predict"])
    with c3:
        st.metric("Best For", cfg["recommended_for"])

    st.divider()
    st.markdown("### 🎛️ Parameter Overrides")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.llm_temperature = st.slider(
            "Temperature Override", 0.0, 1.0,
            float(cfg["temperature"]), 0.05,
            help="Override the model default. Keep low (0.1-0.2) for structured JSON."
        )
    with c2:
        st.session_state.llm_focus = st.selectbox(
            "Focus Area (adds emphasis in prompt)",
            ["All", "Temperature Monitoring", "Fault Handling",
             "OBD Diagnostics", "Safety / ISO 26262", "Timing & Cycles"]
        )

    # Recommended model guide
    st.divider()
    st.markdown("### 📊 Model Comparison Guide")
    st.dataframe(pd.DataFrame([
        {"Model": m, "Size": m.split(":")[1] if ":" in m else "local",
         "Best For": SUPPORTED_MODELS[m]["recommended_for"],
         "Temp": SUPPORTED_MODELS[m]["temperature"],
         "Tokens": SUPPORTED_MODELS[m]["num_predict"]}
        for m in model_options
    ]), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📝 Prompt Engineering")
    st.text_area("System Prompt (read-only — edit in llm/generator.py)", height=140, value=
        "You are an expert automotive validation engineer specializing in BMS.\n"
        "Generate precise bilingual DE+EN test cases.\n"
        "Respond with valid JSON array ONLY — no markdown, no preamble.\n"
        "Follow ISO 26262 functional safety principles.")

    with st.expander("🔍 View Few-Shot Example"):
        st.code("""REQUIREMENT:
BMW_t_HvsCellMinObd = BMW_t_HvsCellCoreMinAct
when QUAL_INT_OK and value in [-40°C, 215°C]
Otherwise → BMW_LIM_MAXERRTEMP_SC

OUTPUT:
[{"TC_ID":"TC_17_8_001","Type_EN":"Nominal","Type_DE":"Normalfall",
  "Objective_EN":"Verify output equals input when qualifier OK",
  "Ziel_DE":"Ausgabe entspricht Eingang wenn Qualifier i.O.",
  "Priority":"High"}]""", language="text")

    st.divider()
    st.markdown("### 📊 RAG Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Context Documents", 1, 10, 3,
                  help="Similar requirements included as LLM context via ChromaDB")
    with c2:
        st.selectbox("Embedding Model",
            ["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"],
            help="Multilingual model recommended for DE+EN requirements")

    if st.button("💾 Save Settings", type="primary"):
        # Save selected model to env for next API restart
        st.success(f"✅ Settings saved! Model: `{sel_model}` · Temp: `{st.session_state.llm_temperature}`")
        st.info(f"💡 To apply model change, restart the API with:\n"
                f"`OLLAMA_MODEL={sel_model} uvicorn backend.api:app --port 8000`")

# ── PAGE: EXPORT ──────────────────────────────────────────────────────────────
def page_export():
    st.title("📤 Export")
    if not st.session_state.tc_run_id and not st.session_state.manual_tcs:
        st.info("Upload requirements first."); return
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("#### 📊 Excel (Complete)")
        st.caption("Unit + ECU + Manual + Edited + Approved — bilingual")
        if st.button("⬇️ Build Excel",use_container_width=True):
            import io
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="xlsxwriter") as w:
                if st.session_state.df_unit is not None:
                    st.session_state.df_unit.to_excel(w,sheet_name="Unit Tests",index=False)
                if st.session_state.df_ecu is not None and len(st.session_state.df_ecu):
                    st.session_state.df_ecu.to_excel(w,sheet_name="ECU Integration",index=False)
                if st.session_state.manual_tcs:
                    pd.DataFrame(st.session_state.manual_tcs).to_excel(w,sheet_name="Manual TCs",index=False)
                if st.session_state.edited_tcs:
                    pd.DataFrame(st.session_state.edited_tcs.values()).to_excel(w,sheet_name="Edited TCs",index=False)
                all_df=pd.concat([
                    st.session_state.df_unit or pd.DataFrame(),
                    st.session_state.df_ecu or pd.DataFrame(),
                    pd.DataFrame(st.session_state.manual_tcs)],ignore_index=True)
                pd.DataFrame([
                    {"Metric":"Total Requirements","Value":len(st.session_state.df_reqs) if st.session_state.df_reqs is not None else 0},
                    {"Metric":"Unit TCs","Value":len(st.session_state.df_unit) if st.session_state.df_unit is not None else 0},
                    {"Metric":"ECU TCs","Value":len(st.session_state.df_ecu) if st.session_state.df_ecu is not None else 0},
                    {"Metric":"Manual TCs","Value":len(st.session_state.manual_tcs)},
                    {"Metric":"Total TCs","Value":len(all_df)},
                    {"Metric":"Approved","Value":len(st.session_state.approved_tcs)},
                    {"Metric":"Exported At","Value":datetime.now().isoformat()},
                ]).to_excel(w,sheet_name="Summary",index=False)
            buf.seek(0)
            st.download_button("📥 Save Excel",data=buf,
                               file_name=f"bms_testcases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        st.markdown("#### 🔧 ECU.TEST Package")
        st.caption("ECU.TEST .pkg.xml + .prj.xml (Tracetronic API)")
        if st.session_state.tc_run_id and st.button("⬇️ ECU.TEST XML",use_container_width=True):
            try:
                r=requests.get(f"{API_URL}/export/{st.session_state.tc_run_id}/canoe_xml",headers=hdrs(),timeout=60)
                st.download_button("📥 Save ECU.TEST XML",data=r.content,
                                   file_name=f"bms_ecutest_{datetime.now().strftime('%Y%m%d')}.xml",mime="application/xml")
            except Exception as e: st.error(str(e))
    with c3:
        st.markdown("#### 📋 Coverage Report")
        st.caption("HTML report with metrics")
        if st.button("ℹ️ How to generate",use_container_width=True):
            st.code("python cicd/generate_report.py")
    st.divider()
    st.code(f"Run ID: {st.session_state.run_id or 'N/A'}\nTC Run ID: {st.session_state.tc_run_id or 'N/A'}\nApproved: {len(st.session_state.approved_tcs)} | Edited: {len(st.session_state.edited_tcs)} | Manual: {len(st.session_state.manual_tcs)}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.token: login_page(); return
    page=sidebar()
    if   "Dashboard"    in page: page_dashboard()
    elif "Upload"       in page: page_upload()
    elif "NLP"          in page: page_nlp()
    elif "Unit"         in page: page_unit_tcs()
    elif "ECU"          in page: page_ecu()
    elif "Edit"         in page: page_edit()
    elif "Manual"       in page: page_manual()
    elif "Evaluation"   in page: page_metrics()
    elif "Chat"         in page: page_chat()
    elif "LLM Settings" in page: page_llm()
    elif "Export"       in page: page_export()

if __name__=="__main__": main()
