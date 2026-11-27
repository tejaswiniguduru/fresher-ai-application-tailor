import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

load_dotenv()


class ResumeChange(BaseModel):
    section: str
    reason: str
    before: str
    after: str


class InterviewWarning(BaseModel):
    missing_skill: str
    honest_interview_tip: str


class KeywordItem(BaseModel):
    keyword: str
    matched: bool
    category: str = "Skill/Tool"


class TailoredResumeContent(BaseModel):
    full_name: str
    contact_line: str
    professional_summary: str
    technical_skills: list[str]
    experience_or_training: list[str]
    projects: list[str]
    education: list[str]
    certifications: list[str] = Field(default_factory=list)


class TailorOutput(BaseModel):
    ats_match_before: int = Field(ge=0, le=100)
    ats_match_after: int = Field(ge=0, le=100)
    jd_keywords: list[KeywordItem] = Field(default_factory=list)
    mistakes_fixed: list[str]
    interview_warnings: list[InterviewWarning]
    changes_made: list[ResumeChange]
    tailored_resume: TailoredResumeContent
    cold_email_subject: str
    cold_email_body: str
    linkedin_dm: str = Field(max_length=280)


SYSTEM_INSTRUCTION = """You are an expert career coach and ATS resume specialist for fresh graduates and early-career candidates.

STRICT RULES AGAINST FABRICATION:
1. Never invent companies, job titles, degrees, dates, internships, certifications, tools, or metrics not present in the user's base resume.
2. You may rewrite phrasing, reorder bullets by relevance to the JD, clarify technical stack terminology already implied by the candidate's work, and highlight transferable fresher strengths (projects, coursework, hackathons, self-learning).
3. If the JD requires a skill or tool missing from the base resume, do NOT claim the candidate has it—put it in `interview_warnings` with an honest way to address it in an interview, and mark `matched: false` in `jd_keywords`.
4. Extract 10 to 16 high-signal technical skills, tools, frameworks, or domain keywords from the Job Description into `jd_keywords`, setting `matched: true` only when truthfully supported by the tailored resume.
5. Keep `linkedin_dm` strictly under 280 characters, natural, concise, and tailored to the target company and role.
6. Return valid JSON matching the requested schema."""


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file or environment variables.")
    return genai.Client(api_key=api_key)


def _select_model(client: genai.Client) -> str:
    preferred = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    try:
        available = set()
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            short = name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                available.add(short)
        for candidate in preferred:
            if candidate in available:
                return candidate
        for short in sorted(available):
            if "gemini" in short and "flash" in short and "embedding" not in short and "image" not in short and "audio" not in short:
                return short
    except Exception:
        pass
    return "gemini-2.5-flash"


def tailor_application(base_resume: str, job_description: str, target_company: str = "Target Company") -> TailorOutput:
    if not base_resume.strip() or not job_description.strip():
        raise ValueError("Both base_resume and job_description are required.")

    client = _get_client()
    model = _select_model(client)

    prompt = f"""Target Company: {target_company}

BASE RESUME:
{base_resume.strip()}

JOB DESCRIPTION:
{job_description.strip()}

Tailor the application honestly and return structured JSON."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=TailorOutput,
                    temperature=0.35,
                ),
            )
            if not response.text:
                raise RuntimeError("Model returned an empty response.")
            return TailorOutput.model_validate_json(response.text)
        except APIError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise