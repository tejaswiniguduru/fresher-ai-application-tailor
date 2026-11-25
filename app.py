from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tailor_engine import tailor_application

app = FastAPI(title="Fresher AI Application Tailor")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


class TailorRequest(BaseModel):
    base_resume: str
    job_description: str
    target_company: str = "Target Company"


@app.get("/", response_class=HTMLResponse)
def home():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/api/tailor")
def api_tailor(req: TailorRequest):
    try:
        result = tailor_application(
            base_resume=req.base_resume,
            job_description=req.job_description,
            target_company=req.target_company,
        )
        return result.model_dump()
    except Exception as exc:
        msg = str(exc)
        status_code = 503 if "503" in msg or "UNAVAILABLE" in msg else 500
        raise HTTPException(status_code=status_code, detail=msg)