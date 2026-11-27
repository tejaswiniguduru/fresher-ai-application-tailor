from io import BytesIO
from pathlib import Path
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from tailor_engine import tailor_application

app = FastAPI(title="Fresher AI Application Tailor")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


class TailorRequest(BaseModel):
    base_resume: str
    job_description: str
    target_company: str = "Target Company"


def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _extract_docx_text(data: bytes) -> str:
    doc = Document(BytesIO(data))
    lines: list[str] = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            lines.append(txt)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def home():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/api/extract-resume")
async def api_extract_resume(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .pdf or .docx resume.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 5 MB size limit.")

    try:
        if filename.endswith(".pdf"):
            extracted = _extract_pdf_text(raw_bytes)
        else:
            extracted = _extract_docx_text(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read file: {exc}")

    if not extracted.strip():
        raise HTTPException(
            status_code=422,
            detail="No selectable text found in the file (scanned image PDFs need OCR). Paste text instead.",
        )

    return {"text": extracted, "filename": file.filename}


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