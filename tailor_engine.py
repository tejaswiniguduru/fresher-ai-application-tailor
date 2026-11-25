import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()


class ResumeChange(BaseModel):
    section: str = Field(description="Section modified, e.g., Projects, Skills, Summary")
    before: str = Field(description="Original phrasing or missing element")
    after: str = Field(description="New tailored phrasing")
    reason: str = Field(description="Why this helps pass ATS or impress the recruiter")


class SkillGapWarning(BaseModel):
    missing_skill: str = Field(description="Skill mentioned in JD but weak/absent in base resume")
    honest_interview_tip: str = Field(
        description="How a fresher can honestly address this in an interview without faking experience"
    )


class TailoredResumeContent(BaseModel):
    full_name: str
    contact_line: str = Field(description="Email | Phone | LinkedIn | GitHub | City")
    professional_summary: str = Field(description="2-3 crisp lines tailored to the target role")
    technical_skills: list[str] = Field(description="Categorized skill lines, e.g., 'Languages: Python, SQL'")
    experience_or_training: list[str] = Field(
        description="Action-oriented bullets for internships, training, or work"
    )
    projects: list[str] = Field(
        description="Strong project bullets highlighting tools, problem solved, and measurable outcome"
    )
    education: list[str] = Field(description="Degree, college, graduation year, CGPA/percentage")
    certifications: list[str] = Field(
        default_factory=list,
        description="Relevant certifications and achievements from the base resume (empty list if none)",
    )


class TailorOutput(BaseModel):
    ats_match_before: int = Field(description="Estimated ATS match score out of 100 before tailoring")
    ats_match_after: int = Field(description="Estimated ATS match score out of 100 after tailoring")
    mistakes_fixed: list[str] = Field(description="3-5 key weaknesses fixed from the original resume")
    changes_made: list[ResumeChange] = Field(
        description="Exact bullet-level edits made so nothing is hidden"
    )
    interview_warnings: list[SkillGapWarning] = Field(
        description="JD skills added or emphasized that the candidate must revise before interviews"
    )
    tailored_resume: TailoredResumeContent
    cold_email_subject: str = Field(description="High-open-rate email subject line for HR or hiring manager")
    cold_email_body: str = Field(description="Crisp 4-5 line cold email tailored to Indian hiring teams")
    linkedin_dm: str = Field(description="Under 280 characters LinkedIn connection note / referral request")


EXCLUDED_MODEL_PARTS = (
    "2.0",
    "2.5",
    "embedding",
    "tts",
    "live",
    "image",
    "transcribe",
    "omni",
    "veo",
    "lyria",
    "banana",
)


def _is_supported_text_model(name: str) -> bool:
    low = name.lower()
    if "gemini" not in low:
        return False
    return not any(part in low for part in EXCLUDED_MODEL_PARTS)


def _get_model_candidates() -> list[str]:
    preferred = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    ]
    available: list[str] = []
    try:
        for m in client.models.list():
            raw_name = getattr(m, "name", "") or ""
            short_name = raw_name.replace("models/", "")
            if short_name and _is_supported_text_model(short_name):
                available.append(short_name)
    except Exception:
        pass

    ordered = [m for m in preferred if not available or m in available]
    for m in available:
        if m not in ordered:
            ordered.append(m)
        if len(ordered) >= 5:
            break

    return ordered or preferred


def tailor_application(
    base_resume: str, job_description: str, target_company: str = "the company"
) -> TailorOutput:
    prompt = f"""
    You are an expert Indian tech recruiter and career coach helping a fresher tailor their job application.

    STRICT RULES:
    1. NEVER invent fake companies, fake job titles, fake years of experience, fake metrics/percentages, or fake contact details (email, phone, LinkedIn, GitHub). If contact info is missing in the base resume, use clear brackets like [Your Email], [Your Phone], [LinkedIn URL], [GitHub URL].
    2. Reframe existing college projects, internships, training, and coursework using strong action verbs and truthful keywords from the Job Description.
    3. Check spelling, spacing, and grammar carefully across every field (no missing spaces or joined words).
    4. Keep the tone practical, sharp, and grounded in Indian hiring norms (no cheesy buzzwords like 'passionate go-getter' or 'rockstar').
    5. Keep the LinkedIn DM strictly under 280 characters so it fits a LinkedIn connection request note.
    6. Flag every skill from the JD that is weak or missing in the original resume under `interview_warnings` so the fresher prepares honestly.

    TARGET COMPANY: {target_company}

    --- BASE RESUME ---
    {base_resume}

    --- JOB DESCRIPTION ---
    {job_description}
    """

    candidates = _get_model_candidates()
    last_error = None

    for model_name in candidates:
        for attempt in range(2):
            try:
                print(f"[tailor_engine] Trying {model_name} (attempt {attempt + 1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TailorOutput,
                        temperature=0.3,
                    ),
                )
                return TailorOutput.model_validate_json(response.text)
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                print(f"[tailor_engine] {model_name} error: {err_str}")
                if "404" in err_str or "NOT_FOUND" in err_str:
                    break
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"All candidate models failed. Last error: {last_error}")


if __name__ == "__main__":
    sample_resume = """
    Tejaswini
    B.Tech CSE (AI & ML), 2025 | Telangana
    Skills: Python, Machine Learning, SQL, Git, Basics of Automation Testing
    Training: Graduate Trainee (3 months) - Worked on test case execution and basic automation scripts.
    Projects:
    1. Fake News Detection using NLP - Built a classifier using Python and Scikit-learn.
    2. Student Chatbot - Created a Q&A bot using basic LLM APIs.
    """

    sample_jd = """
    Role: Junior GenAI / Python Engineer
    Requirements:
    - Strong Python programming and API integration skills
    - Hands-on experience with LLMs, prompt engineering, RAG pipelines, or AI agents
    - Familiarity with FastAPI or Flask, JSON handling, and Git
    - Good problem-solving and communication skills
    """

    print("Tailoring application... please wait a few seconds...\n")
    result = tailor_application(sample_resume, sample_jd, target_company="Acme AI Labs")
    print(json.dumps(result.model_dump(), indent=2))