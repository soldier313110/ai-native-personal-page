import json
import os
from typing import List, Optional, Literal
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(
    title="Mohammad Mehdi Sahraei - AI-Native Personal Page",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROFILE_PATH = Path(__file__).parent / "profile.json"

def load_full_profile() -> dict:
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def filter_public_records(data):
    """Recursively strips out any dictionary where 'public' is explicitly False."""
    if isinstance(data, dict):
        if data.get("public") is False:
            return "[REDACTED_NON_PUBLIC_FIELD]"
        return {k: filter_public_records(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [filter_public_records(item) for item in data]
    return data

async def fetch_github_data(query: str) -> dict:
    """Fetches live repository & profile data from GitHub REST API."""
    results = {"service": "GitHub REST API (api.github.com)", "data": {}}
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Sahraei-AI-Page"}
    
    repos_to_check = []
    q_lower = query.lower()
    if any(k in q_lower for k in ["پینگ", "ping", " بازی", "game", "c ", "سی"]):
        repos_to_check.append("soldier313110/pingpong")
    if any(k in q_lower for k in ["هوشمند", "smart", "home", "خانه", "جاوا", "java", "ap", "پیشرفته"]):
        repos_to_check.append("Sharif-AP/Smart-Home-Automation-System")
    if not repos_to_check:
        repos_to_check = ["soldier313110/pingpong", "Sharif-AP/Smart-Home-Automation-System"]

    async with httpx.AsyncClient(timeout=6.0) as client:

        try:
            user_resp = await client.get("https://api.github.com/users/soldier313110", headers=headers)
            if user_resp.status_code == 200:
                u = user_resp.json()
                results["data"]["github_user"] = {
                    "login": u.get("login"),
                    "public_repos": u.get("public_repos"),
                    "followers": u.get("followers"),
                    "profile_url": u.get("html_url"),
                    "created_at": u.get("created_at")
                }
        except Exception as e:
            results["data"]["user_error"] = str(e)

        results["data"]["repositories"] = {}
        for repo in repos_to_check:
            try:
                repo_resp = await client.get(f"https://api.github.com/repos/{repo}", headers=headers)
                if repo_resp.status_code == 200:
                    r = repo_resp.json()
                    results["data"]["repositories"][repo] = {
                        "full_name": r.get("full_name"),
                        "description": r.get("description"),
                        "language": r.get("language"),
                        "stargazers_count": r.get("stargazers_count"),
                        "forks_count": r.get("forks_count"),
                        "open_issues_count": r.get("open_issues_count"),
                        "updated_at": r.get("updated_at"),
                        "html_url": r.get("html_url")
                    }
            except Exception as e:
                results["data"]["repositories"][repo] = {"error": str(e)}

    return results

async def fetch_openlibrary_books(topic: str = "software engineering") -> dict:
    """Fetches real book recommendations from Open Library API based on user courses/projects."""
    results = {"service": "Open Library API (openlibrary.org)", "topic_queried": topic, "books": []}
    async with httpx.AsyncClient(timeout=6.0) as client:
        try:
            resp = await client.get(
                "https://openlibrary.org/search.json",
                params={"q": topic, "limit": 4, "fields": "title,author_name,first_publish_year,subject"}
            )
            if resp.status_code == 200:
                docs = resp.json().get("docs", [])
                for d in docs[:4]:
                    results["books"].append({
                        "title": d.get("title"),
                        "authors": d.get("author_name", ["Unknown"])[:2],
                        "first_publish_year": d.get("first_publish_year")
                    })
        except Exception as e:
            results["error"] = str(e)
    return results

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    persona_mode: Optional[str] = Field(default="standard", description="standard | recruiter | technical")

class AIResponseSchema(BaseModel):
    answer: str
    capability_used: Literal[
        "answer_from_record",
        "answer_from_external_api",
        "ask_clarifying_question",
        "refuse_unpublished_facts",
        "say_out_of_scope",
        "job_fit_evaluation",
    ]
    intent_detected: Literal[
        "project_summary",
        "reading_suggestion",
        "job_fit_evaluation",
        "general_record_query",
        "ambiguous_request",
        "unpublished_or_private_fact",
        "out_of_scope_or_prohibited",
    ]
    fields_read: List[str]
    external_api_used: Optional[str]
    external_api_details: Optional[dict] = None
    missing_or_refused_facts: List[str]
    reasoning_trace: str
    confidence_or_fit_score: Optional[int] = Field(
        default=None,
        description="If intent is job_fit_evaluation, a score from 0 to 100 based strictly on record."
    )


SYSTEM_PROMPT = """You are the official AI-Native Personal Page Agent for **Mohammad Mehdi Sahraei** (محمد مهدی صحرایی).
Your behavior is strictly governed by the provided PUBLIC RECORD and EXTERNAL API DATA.

### ALLOWED CAPABILITIES (You MUST pick exactly ONE for `capability_used`):
1. `answer_from_record`: When answering strictly using the provided JSON record (e.g., summarizing a project, evaluating job fit, listing skills/courses).
2. `answer_from_external_api`: When your answer incorporates live data fetched from the External APIs (GitHub REST API or Open Library API) alongside the record.
3. `ask_clarifying_question`: When the visitor's request is ambiguous, underspecified, or unclear (e.g., "پروژه‌ات چطوره؟" without naming which project, or "آیا برای کار ما مناسبی؟" without describing the job requirements).
4. `refuse_unpublished_facts`: When the visitor asks for facts NOT in the record or marked as non-public/unpublished (e.g., phone number, email, GPA, grades other than Advanced Programming, salary, age, exact address, or unrecorded skills like Rust/React).
5. `say_out_of_scope`: When the visitor asks you to perform prohibited actions (send an email, book a meeting/interview, speak on behalf of Sharif University/Municipality/Hoze Honari, or discuss politics/unrelated topics).

### REQUIRED REQUEST TYPES TO SUPPORT:
- **Summarize a project (`project_summary`)**: Summarize Bargino, Smart-Home-Automation-System, or pingpong using ONLY the record and GitHub API data. If the user doesn't specify which project and asks about "the project", use `ask_clarifying_question`.
- **Suggest reading (`reading_suggestion`)**: Suggest books/topics to read based on Mohammad Mehdi's courses (Advanced Programming - Java/OOP) and projects (Python, Java, C, Game Dev, Smart Home, Data Analysis), enriched by the Open Library API books provided. Always explicitly name the external service ("Open Library API") when used.
- **Job Fit Check (`job_fit_evaluation`)**: Evaluate whether Mohammad Mehdi fits a job description provided by the visitor. Compare required skills vs. recorded skills/projects/courses. Explicitly list matching facts AND missing/unverified facts. Provide a realistic `confidence_or_fit_score` (0-100). If no job description or role criteria are given, use `ask_clarifying_question`.

### STRICT GUARDRAILS (ZERO HALLUCINATION POLICY):
- NEVER invent facts, courses, grades, frameworks, or experiences not explicitly in the PUBLIC RECORD or EXTERNAL API DATA.
- NEVER reveal private fields (`phone_number`, `email_address`, `home_address`, `overall_gpa`).
- If a user asks part-valid and part-unpublished questions, answer ONLY the valid part from the record and explicitly list the refused part in `missing_or_refused_facts`.
- Always respond in fluent, polite **Persian (فارسی)** in the `answer` and `reasoning_trace` fields.

### OUTPUT FORMAT:
Return ONLY a valid JSON object matching the exact schema requested:
{
  "answer": "پاسخ کامل و دقیق به زبان فارسی...",
  "capability_used": "answer_from_record | answer_from_external_api | ask_clarifying_question | refuse_unpublished_facts | say_out_of_scope",
  "intent_detected": "project_summary | reading_suggestion | job_fit_evaluation | general_record_query | ambiguous_request | unpublished_or_private_fact | out_of_scope_or_prohibited",
  "fields_read": ["records.projects.items[0]", "records.skills.technical", ...],
  "external_api_used": "GitHub REST API (api.github.com)" or "Open Library API (openlibrary.org)" or null,
  "missing_or_refused_facts": ["معدل کل (منتشر نشده)", ...],
  "reasoning_trace": "توضیح گام‌به‌گام به فارسی که کدام فیلدها بررسی شد، چرا این قابلیت انتخاب شد و چه مرزهایی رعایت گردید.",
  "confidence_or_fit_score": null or integer (0-100)
}
"""

@app.post("/api/ask", response_model=AIResponseSchema)
async def ask_agent(req: AskRequest):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GAPGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="کلید API تنظیم نشده است. لطفاً متغیر محیطی LLM_API_KEY را تنظیم کنید."
        )

    full_profile = load_full_profile()
    public_profile = filter_public_records(full_profile)

    q_lower = req.question.lower()
    external_context = {}

    if any(w in q_lower for w in ["پروژه", "گیت", "پینگ", "هوشمند", "برگینو", "کد", "مخزن", "ریپو", "ستاره", "github", "project", "repo", "smart", "pingpong", "خلاصه"]):
        external_context["github_live_data"] = await fetch_github_data(req.question)

    if any(w in q_lower for w in ["کتاب", "مطالعه", "بخونم", "بخوانم", "منبع", "منابع", "پیشنهاد", "درس", "read", "book", "suggest", "study"]):
        topic = "Java Object Oriented Programming"
        if "پایتون" in q_lower or "python" in q_lower or "داده" in q_lower:
            topic = "Python Data Analysis"
        elif "بازی" in q_lower or "پینگ" in q_lower or " c " in q_lower:
            topic = "C Programming Game Development"
        external_context["openlibrary_live_data"] = await fetch_openlibrary_books(topic)

    if not external_context:
        external_context["github_live_data"] = await fetch_github_data(req.question)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    user_prompt = f"""
### PUBLIC RECORD (JSON):
{json.dumps(public_profile, ensure_ascii=False, indent=2)}

### LIVE EXTERNAL API DATA:
{json.dumps(external_context, ensure_ascii=False, indent=2)}

### VISITOR PERSONA MODE:
{req.persona_mode}

### VISITOR QUESTION:
"{req.question}"

Analyze the visitor question strictly against the PUBLIC RECORD and LIVE EXTERNAL API DATA, and return the JSON response.
"""

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = completion.choices[0].message.content
        parsed = json.loads(raw_content)
        parsed["external_api_details"] = external_context
        return AIResponseSchema(**parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش هوش مصنوعی: {str(e)}")

@app.get("/api/record")
async def get_public_record():
    """Returns the full profile JSON (with schema and public markers) for inspection."""
    return load_full_profile()

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Backend is running! Create static/index.html for the UI."}
