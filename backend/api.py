"""
backend/api.py
FastAPI REST Backend for BMS GenAI Assistant
Run with: uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import hashlib
import hmac
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger


def numpy_safe(obj):
    """
    Recursively convert numpy/pandas scalar types to native Python so
    FastAPI's JSON serializer never chokes on int64 / float64 / bool_.
    """
    if isinstance(obj, dict):
        return {k: numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Auth
from jose import JWTError, jwt

# Internal modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.ingestion import BMSRequirementLoader
from backend.database import BMSDatabase
from nlp.pipeline import BMSNLPPipeline
from nlp.test_generator import BMSTestCaseGenerator
from llm.generator import BMSLLMGenerator

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY     = os.getenv("SECRET_KEY", "bms-secret-key-2024")
ALGORITHM      = "HS256"
TOKEN_EXPIRE   = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 480))
UPLOADS_DIR    = Path(os.getenv("UPLOADS_DIR", "./data/uploads"))
OUTPUTS_DIR    = Path(os.getenv("OUTPUTS_DIR", "./outputs"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth setup (hashlib — no bcrypt version issues) ───────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def _hash_password(password: str) -> str:
    """Hash password using SHA-256 + HMAC — no bcrypt dependency."""
    return hmac.new(SECRET_KEY.encode(), password.encode(), hashlib.sha256).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_password(plain), hashed)

# Demo users — passwords hashed at import time (no bcrypt needed)
USERS_DB = {
    "engineer": {
        "username": "engineer",
        "hashed_password": _hash_password("bms2024"),
        "role": "engineer",
    },
    "admin": {
        "username": "admin",
        "hashed_password": _hash_password("admin2024"),
        "role": "admin",
    },
}

def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or username not in USERS_DB:
            raise HTTPException(status_code=401, detail="Invalid token")
        return USERS_DB[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BMS GenAI Assistant API",
    description="Automated test case generation for BMS ECU requirements",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded heavy modules
_nlp_pipeline: Optional[BMSNLPPipeline] = None
_tc_generator: Optional[BMSTestCaseGenerator] = None
_llm_generator: Optional[BMSLLMGenerator] = None
_db: Optional[BMSDatabase] = None
_loader = BMSRequirementLoader()

def get_nlp():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = BMSNLPPipeline()
    return _nlp_pipeline

def get_tc_gen():
    global _tc_generator
    if _tc_generator is None:
        _tc_generator = BMSTestCaseGenerator()
    return _tc_generator

def get_llm():
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = BMSLLMGenerator(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            chroma_path=os.getenv("CHROMA_PATH", "./data/chroma_db"),
        )
    return _llm_generator

def get_db():
    global _db
    if _db is None:
        _db = BMSDatabase(os.getenv("DB_PATH", "./data/bms.db"))
    return _db

# ── Pydantic models ───────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class ApproveRequest(BaseModel):
    tc_ids: List[str]
    approved_by: str

# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=Token, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": form.username})
    return Token(
        access_token=token,
        token_type="bearer",
        username=form.username,
        role=user["role"],
    )

# ── Requirements endpoints ────────────────────────────────────────────────────

@app.post("/requirements/upload", tags=["Requirements"])
async def upload_requirements(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """Upload an Excel requirements file and run NLP pipeline."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Only Excel files supported")

    # Save file
    path = UPLOADS_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Uploaded: {file.filename}")

    # Load & validate
    df = _loader.load_excel(path)
    validation = _loader.validate(df)
    if not validation["valid"]:
        logger.warning(f"Validation issues: {validation['issues']}")

    # Run NLP
    nlp = get_nlp()
    df_enriched = nlp.run(df)

    # Save to DB
    db = get_db()
    run_id = db.save_requirements(df_enriched, file.filename)

    # Index in ChromaDB
    llm = get_llm()
    llm.index_requirements(df_enriched.to_dict("records"))

    return numpy_safe({
        "status": "success",
        "run_id": run_id,
        "file": file.filename,
        "validation": validation,
        "requirements_count": len(df_enriched),
        "critical_count": int(df_enriched["is_critical"].sum()),
    })

@app.get("/requirements/{run_id}", tags=["Requirements"])
def get_requirements(run_id: str, current_user=Depends(get_current_user)):
    """Get enriched requirements for a run."""
    db = get_db()
    data = db.get_requirements(run_id)
    if data is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return numpy_safe(data)

@app.get("/requirements/{run_id}/stats", tags=["Requirements"])
def get_stats(run_id: str, current_user=Depends(get_current_user)):
    """Get NLP statistics for a run."""
    db = get_db()
    return numpy_safe(db.get_stats(run_id))

# ── Test case generation endpoints ───────────────────────────────────────────

@app.post("/testcases/generate/{run_id}", tags=["Test Cases"])
def generate_test_cases(
    run_id: str,
    use_llm: bool = False,
    current_user=Depends(get_current_user),
):
    """Generate unit + ECU integration test cases for a run."""
    db = get_db()
    df = db.get_requirements_df(run_id)
    if df is None:
        raise HTTPException(404, f"Run {run_id} not found")

    tc_gen = get_tc_gen()
    df_unit, df_ecu = tc_gen.run(df)

    # Optional LLM enhancement
    llm_tcs = []
    if use_llm:
        llm = get_llm()
        if llm.is_available():
            for _, row in df.iterrows():
                tcs = llm.generate_test_cases(row.to_dict())
                llm_tcs.extend(tcs)
            logger.info(f"LLM generated {len(llm_tcs)} additional TCs")

    tc_id = db.save_test_cases(run_id, df_unit, df_ecu, llm_tcs)

    return numpy_safe({
        "status": "success",
        "run_id": run_id,
        "tc_run_id": tc_id,
        "unit_tcs": len(df_unit),
        "ecu_tcs": len(df_ecu),
        "llm_tcs": len(llm_tcs),
        "total": len(df_unit) + len(df_ecu) + len(llm_tcs),
    })

@app.get("/testcases/{tc_run_id}", tags=["Test Cases"])
def get_test_cases(tc_run_id: str, current_user=Depends(get_current_user)):
    """Retrieve generated test cases."""
    db = get_db()
    return db.get_test_cases(tc_run_id)

@app.post("/testcases/{tc_run_id}/approve", tags=["Test Cases"])
def approve_test_cases(
    tc_run_id: str,
    req: ApproveRequest,
    current_user=Depends(get_current_user),
):
    """Mark test cases as approved by an engineer."""
    db = get_db()
    db.approve_test_cases(tc_run_id, req.tc_ids, req.approved_by)
    return {"status": "approved", "count": len(req.tc_ids)}

# ── Export endpoints ──────────────────────────────────────────────────────────

@app.get("/export/{tc_run_id}/excel", tags=["Export"])
def export_excel(tc_run_id: str, current_user=Depends(get_current_user)):
    """Export test cases to Excel."""
    db = get_db()
    path = db.export_excel(tc_run_id, OUTPUTS_DIR)
    if not path:
        raise HTTPException(500, "Export failed")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=Path(path).name,
    )

@app.get("/export/{tc_run_id}/ecutest", tags=["Export"])
def export_ecutest(
    tc_run_id: str,
    ecutest_path: str = "C:/Program Files/TraceTronic/ECU-TEST",
    current_user=Depends(get_current_user),
):
    """Generate ECU.test .pkg packages + .prj project + REST runner."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ecu.ecutest_integration import BMSECUTestSuiteGenerator
    import zipfile

    db = get_db()
    data = db.get_test_cases(tc_run_id)
    if not data:
        raise HTTPException(404, f"TC run {tc_run_id} not found")

    ecu_tcs = data.get("ecu_tcs", [])
    if not ecu_tcs:
        raise HTTPException(400, "No ECU integration test cases found")

    df_ecu = pd.DataFrame(ecu_tcs)
    suite_dir = OUTPUTS_DIR / f"ecutest_{tc_run_id}"
    suite_gen = BMSECUTestSuiteGenerator(ecutest_install_path=ecutest_path)
    result = suite_gen.generate_suite(df_ecu, str(suite_dir))

    # Zip everything for download
    zip_path = OUTPUTS_DIR / f"ecutest_suite_{tc_run_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for pkg in result["packages"]:
            zf.write(pkg, f"Packages/{Path(pkg).name}")
        for f in [result["project"], result["tbc"], result["tcf"], result["runner"]]:
            zf.write(f, Path(f).name)

    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"bms_ecutest_suite_{tc_run_id}.zip")
    """Export ECU test cases as CANoe XML."""
    db = get_db()
    path = db.export_canoe_xml(tc_run_id, OUTPUTS_DIR)
    if not path:
        raise HTTPException(500, "Export failed")
    return FileResponse(path, media_type="application/xml", filename=Path(path).name)

# ── Chat / RAG endpoint ───────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest, current_user=Depends(get_current_user)):
    """Ask a question about the BMS requirements using RAG + LLM."""
    import uuid
    llm = get_llm()
    answer = llm.answer_question(req.question)
    session_id = req.session_id or str(uuid.uuid4())
    return ChatResponse(answer=answer, session_id=session_id)

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    llm = get_llm()
    return {
        "status": "ok",
        "version": "1.0.0",
        "ollama_available": llm.is_available(),
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/", tags=["System"])
def root():
    return {
        "app": "BMS GenAI Assistant",
        "docs": "/docs",
        "health": "/health",
    }
