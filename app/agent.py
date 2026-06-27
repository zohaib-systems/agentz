# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from groq import Groq
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from rapidfuzz import fuzz
from google.adk.models import Gemini
from google.genai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


load_dotenv()

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

session_service = InMemorySessionService()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DASHBOARD_FILE = "dashboard_jobs.json"
FETCH_INTERVAL_HOURS = int(os.getenv("FETCH_INTERVAL_HOURS", "4"))
FUZZY_THRESHOLD = 75
GROQ_MODEL = "llama-3.3-70b-versatile"

# Google Calendar
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_TOKEN = "token.pickle"
CALENDAR_CREDS = "credentials.json"

# ---------------------------------------------------------------------------
# Groq client (for proposal drafting)
# ---------------------------------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

INJECTION_PHRASES = [
    "ignore previous instructions",
    "bypass",
    "pretend you are",
    "disregard",
    "jailbreak",
    "ignore all rules",
]


def security_screen(text: str) -> str | None:
    """Return an error string if a prompt-injection attempt is detected."""
    text_lower = text.lower()
    for phrase in INJECTION_PHRASES:
        if phrase in text_lower:
            return (
                f"⚠️ Security: Suspicious input detected ('{phrase}'). Request blocked."
            )
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_keywords() -> list[str]:
    """Load keyword list from skillset.md."""
    try:
        with open("context/skillset.md", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    keywords = []
    if "Keywords to Match in Job Posts" in content:
        parts = content.split("Keywords to Match in Job Posts")
        if len(parts) > 1:
            section = parts[1].split("##")[0].strip()
            cleaned = section.replace("-", "").replace("\n", " ")
            keywords = [k.strip() for k in cleaned.split(",") if k.strip()]
    return keywords


def _fuzzy_match_keywords(job_text: str, keywords: list[str]) -> list[str]:
    """Return keywords that fuzzy-match against the job text."""
    matched = []
    job_lower = job_text.lower()

    for kw in keywords:
        kw_lower = kw.lower()
        words_in_kw = kw_lower.split()

        if len(words_in_kw) == 1:
            job_tokens = re.findall(r"\b\w+\b", job_lower)
            best_score = max(
                (fuzz.partial_ratio(kw_lower, tok) for tok in job_tokens),
                default=0,
            )
        else:
            best_score = fuzz.token_set_ratio(kw_lower, job_lower)

        if best_score >= FUZZY_THRESHOLD:
            matched.append(kw)

    return matched


def _compute_score(matched: list[str], keywords: list[str], job_text: str) -> int:
    """Compute a 0-100 score with a floor rule for core skills."""
    total = len(keywords) if keywords else 1
    raw = int((len(matched) / total) * 100)

    # Bonus points per matched keyword (each match is worth more)
    bonus_score = min(len(matched) * 12, 100)

    # Take the higher of ratio-based or bonus-based score
    score = max(raw, bonus_score)

    # Floor rule — only if no meaningful matches found
    floor_keywords = [
        "react",
        "node",
        "nodejs",
        "python",
        "medical",
        "ai",
        "mern",
        "next.js",
        "nextjs",
        "typescript",
        "mongodb",
        "firebase",
    ]
    job_lower = job_text.lower()
    if (
        any(fuzz.partial_ratio(w, job_lower) >= FUZZY_THRESHOLD for w in floor_keywords)
        and score < 60
    ):
        score = 60

    return min(score, 100)


def _load_dashboard() -> list[dict]:
    """Load existing dashboard jobs from JSON file."""
    if not os.path.exists(DASHBOARD_FILE):
        return []
    try:
        with open(DASHBOARD_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_dashboard(jobs: list[dict]) -> None:
    """Persist dashboard jobs to JSON file."""
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


def _upsert_dashboard_job(job: dict) -> None:
    """Add or update a job in the dashboard file (deduped by job_id)."""
    jobs = _load_dashboard()
    existing_ids = {j.get("job_id") for j in jobs}
    if job.get("job_id") not in existing_ids:
        jobs.append(job)
        jobs.sort(key=lambda j: j.get("score", 0), reverse=True)
        _save_dashboard(jobs)


def _get_calendar_service():
    """Get authenticated Google Calendar service."""
    creds = None
    if Path(CALENDAR_TOKEN).exists():
        with open(CALENDAR_TOKEN, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CALENDAR_CREDS, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(CALENDAR_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Google Jobs Scraper via SerpAPI
# ---------------------------------------------------------------------------


def _fetch_google_jobs(query: str = "React developer AI freelance") -> list[dict]:
    """Fetch job listings from Google Jobs via SerpAPI."""
    if not SERPAPI_KEY:
        return [{"error": "SERPAPI_KEY not set in .env"}]

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 10,
        "chips": "date_posted:week",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]

    jobs_raw = data.get("jobs_results", [])
    jobs = []
    for j in jobs_raw:
        description = j.get("description", "")
        highlights = j.get("job_highlights", [])
        for h in highlights:
            for item in h.get("items", []):
                description += f" {item}"

        jobs.append(
            {
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("location", ""),
                "description": description.strip(),
                "apply_link": j.get("related_links", [{}])[0].get("link", ""),
                "fetched_at": datetime.utcnow().isoformat(),
            }
        )

    return jobs


# ---------------------------------------------------------------------------
# Core auto-fetch + score pipeline
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "React developer AI freelance remote",
    "MERN stack developer freelance remote",
    "Next.js developer freelance",
    "AI chatbot developer freelance",
    "Node.js backend developer remote freelance",
    "healthcare AI developer freelance",
]


def run_auto_fetch_and_score() -> dict:
    """Full pipeline: fetch → score → persist to dashboard."""
    keywords = _load_keywords()
    all_jobs: list[dict] = []

    for query in SEARCH_QUERIES:
        fetched = _fetch_google_jobs(query)
        all_jobs.extend(fetched)
        time.sleep(0.5)

    seen = set()
    unique_jobs = []
    for job in all_jobs:
        key = (job.get("title", "").lower(), job.get("company", "").lower())
        if "error" not in job and key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    scored = []
    for idx, job in enumerate(unique_jobs):
        blob = f"{job['title']} {job['description']}"
        matched = _fuzzy_match_keywords(blob, keywords)
        score = _compute_score(matched, keywords, blob)

        dashboard_entry = {
            "job_id": f"{job['company']}_{idx}_{int(time.time())}",
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "apply_link": job["apply_link"],
            "score": score,
            "matched_skills": matched,
            "reasoning": (
                "Strong match"
                if score >= 80
                else "Partial match"
                if score >= 50
                else "Weak match"
            ),
            "fetched_at": job["fetched_at"],
            "status": "new",
        }

        _upsert_dashboard_job(dashboard_entry)
        scored.append(dashboard_entry)

    high_quality = [j for j in scored if int(j.get("score", 0)) >= 70]
    return {
        "total_fetched": len(unique_jobs),
        "total_scored": len(scored),
        "high_quality": len(high_quality),
        "dashboard_file": DASHBOARD_FILE,
        "top_3": [
            {"title": j["title"], "company": j["company"], "score": j["score"]}
            for j in scored[:3]
        ],
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def auto_fetch_jobs(tool_context: ToolContext) -> str:
    """Trigger a full Google Jobs fetch + score cycle and push results to dashboard."""
    result = run_auto_fetch_and_score()

    if "error" in result:
        return json.dumps(result)

    tool_context.state["last_fetch"] = datetime.utcnow().isoformat()
    tool_context.state["last_fetch_summary"] = result

    return (
        f"✅ Auto-fetch complete!\n"
        f"→ Fetched: {result['total_fetched']} unique jobs\n"
        f"→ Scored:  {result['total_scored']} jobs\n"
        f"→ High quality (≥70): {result['high_quality']} jobs\n"
        f"→ Dashboard updated: {DASHBOARD_FILE}\n\n"
        f"Top 3:\n"
        + "\n".join(
            f"  {i + 1}. {j['title']} @ {j['company']} — Score: {j['score']}"
            for i, j in enumerate(result.get("top_3", []))
        )
    )


def score_opportunity(job_description: str, tool_context: ToolContext) -> str:
    """Score a manually pasted job description against Zohaib's skillset."""
    block = security_screen(job_description)
    if block:
        return json.dumps({"error": block, "score": 0})

    keywords = _load_keywords()
    matched = _fuzzy_match_keywords(job_description, keywords)
    score = _compute_score(matched, keywords, job_description)

    if "scored_jobs" not in tool_context.state:
        tool_context.state["scored_jobs"] = []
    scored_list = list(tool_context.state["scored_jobs"])
    scored_list.append({"job": job_description[:100], "score": score})
    tool_context.state["scored_jobs"] = scored_list

    return json.dumps(
        {
            "score": score,
            "matched_skills": matched,
            "reasoning": (
                "Strong match"
                if score >= 80
                else "Partial match"
                if score >= 50
                else "Weak match"
            ),
        }
    )


def draft_proposal(job_description: str, score: str, tool_context: ToolContext) -> str:
    """Draft a proposal in Zohaib's tone using Groq and queue for human review."""
    block = security_screen(job_description)
    if block:
        return block

    try:
        score_val = float(score)
    except ValueError:
        score_val = 0

    if score_val < 50:
        return "Job score too low to draft proposal."

    try:
        with open("context/skillset.md", encoding="utf-8") as f:
            skillset = f.read()
    except Exception:
        skillset = "Error reading skillset."

    prompt = (
        "Draft a freelance job proposal for the job description below. "
        "Lead with Zohaib's relevant past work. "
        "Be direct and confident — no fluff, no filler phrases like "
        "'I am passionate about'. Write in first person. "
        "Use short paragraphs and specific deliverables.\n\n"
        f"Skillset & Tone Guidelines:\n{skillset}\n\n"
        f"Job Description:\n{job_description}"
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        proposal_text = response.choices[0].message.content
    except Exception as e:
        proposal_text = f"Error generating proposal: {e}"

    if "proposals_drafted" not in tool_context.state:
        tool_context.state["proposals_drafted"] = []
    proposals = list(tool_context.state["proposals_drafted"])
    proposals.append(
        {
            "job_description": job_description[:100],
            "proposal": proposal_text,
            "status": "pending_review",
        }
    )
    tool_context.state["proposals_drafted"] = proposals

    return (
        f"📋 PROPOSAL READY FOR REVIEW\n"
        f"Job Score: {score}\n\n"
        f"{proposal_text}\n\n"
        f"─────────────────────────────────\n"
        f"⚠️  Human review required before sending.\n"
        f"Copy the proposal above and paste it manually on Fiverr/Upwork."
    )


def get_morning_briefing(tool_context: ToolContext) -> str:
    """Return a morning briefing from LM-OS data."""
    if "briefing_shown" not in tool_context.state:
        tool_context.state["briefing_shown"] = False

    try:
        with open("context/lmos_data.json", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading LM-OS data: {e}"

    goals = data.get("goals", [])
    top_goal = goals[0] if goals else {}
    goal_title = top_goal.get("title", "None")
    goal_deadline = top_goal.get("deadline", "None")

    if tool_context.state["briefing_shown"]:
        return (
            f"Briefing already shown this session. "
            f"Quick recap → Top goal: {goal_title} by {goal_deadline}."
        )

    tool_context.state["briefing_shown"] = True
    surplus = data.get("finances", {}).get("monthly_surplus_pkr", 0)

    habits = data.get("habits", [])
    active_habit = next((h for h in habits if h.get("status") == "active"), {})
    habit_str = (
        f"{active_habit.get('name', 'None')} "
        f"(Day {active_habit.get('streak', 0)} of {active_habit.get('target_days', 0)})"
        if active_habit
        else "None"
    )

    deadlines = data.get("deadlines", [])
    deadlines_str = ", ".join(deadlines) if deadlines else "No deadlines today"

    dashboard_jobs = _load_dashboard()
    new_jobs = [j for j in dashboard_jobs if j.get("status") == "new"]
    high_score_new = [j for j in new_jobs if j.get("score", 0) >= 70]

    jobs_line = (
        f"💼 New Jobs in Dashboard: {len(new_jobs)} "
        f"({len(high_score_new)} high-quality ≥70)"
        if new_jobs
        else "💼 No new jobs in dashboard"
    )

    return (
        f"🌅 Good morning, Zohaib!\n\n"
        f"🎯 Top Goal: {goal_title} (Deadline: {goal_deadline})\n"
        f"💰 Monthly Surplus: {surplus:,} PKR\n"
        f"💪 Active Habit: {habit_str}\n"
        f"⏰ Deadlines: {deadlines_str}\n"
        f"{jobs_line}"
    )


def triage_emails(emails: str) -> str:
    """Categorise emails by importance."""
    results = []
    for email in emails.split(","):
        email = email.strip()
        if not email:
            continue

        parts = email.split("|")
        subject = parts[0].strip().lower() if parts else ""
        sender = parts[1].strip().lower() if len(parts) > 1 else ""

        category = "IGNORE"
        reason = "Does not match any important category."

        if any(w in subject for w in ["invoice", "receipt", "payment", "paid"]):
            category = "PAYMENT"
            reason = "Payment related."
        elif any(w in sender for w in ["client", "healthclinic", "project"]) or any(
            w in subject for w in ["project", "update", "revision", "delivery"]
        ):
            category = "CLIENT"
            reason = "From a client or project-related."
        elif any(
            w in subject for w in ["order", "inquiry", "job", "new gig", "request"]
        ) or any(w in sender for w in ["fiverr", "upwork", "linkedin"]):
            category = "OPPORTUNITY"
            reason = "Potential new opportunity."

        results.append({"email": email, "category": category, "reason": reason})

    return json.dumps(results, indent=2)


def get_dashboard_summary(tool_context: ToolContext) -> str:
    """Return a summary of current dashboard jobs by status and score."""
    jobs = _load_dashboard()

    if not jobs:
        return "Dashboard is empty. Run 'fetch jobs' to populate it."

    by_status: dict[str, list] = {}
    for j in jobs:
        s = j.get("status", "new")
        by_status.setdefault(s, []).append(j)

    high = [j for j in jobs if j.get("score", 0) >= 70]
    mid = [j for j in jobs if 50 <= j.get("score", 0) < 70]
    low = [j for j in jobs if j.get("score", 0) < 50]

    top_jobs = "\n".join(
        f"  • {j['title']} @ {j['company']} — Score {j['score']} [{j.get('status', 'new')}]"
        for j in jobs[:5]
    )

    last_fetch = tool_context.state.get("last_fetch", "Not fetched this session")

    return (
        f"📊 Dashboard Summary\n"
        f"────────────────────\n"
        f"Total jobs: {len(jobs)}\n"
        f"  🟢 High match (≥70): {len(high)}\n"
        f"  🟡 Mid  match (50-69): {len(mid)}\n"
        f"  🔴 Low  match (<50): {len(low)}\n\n"
        f"By status:\n"
        + "\n".join(f"  {k}: {len(v)}" for k, v in by_status.items())
        + f"\n\nTop 5 Jobs:\n{top_jobs}\n\n"
        f"Last fetch: {last_fetch}"
    )


# ---------------------------------------------------------------------------
# Calendar Tools
# ---------------------------------------------------------------------------


def schedule_event(
    title: str,
    date: str,
    time: str,
    duration_minutes: int,
    description: str,
    tool_context: ToolContext,
) -> str:
    """Schedule an event in Google Calendar.

    Args:
        title: Event title
        date: Date in YYYY-MM-DD format
        time: Time in HH:MM format (24hr)
        duration_minutes: Duration in minutes
        description: Event description or notes
        tool_context: Tool context
    Returns: Confirmation string with event details
    """
    try:
        service = _get_calendar_service()
        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Karachi"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Karachi"},
            "reminders": {
                "useDefault": False,
                "overrides": [{"method": "popup", "minutes": 30}],
            },
        }

        created = service.events().insert(calendarId="primary", body=event).execute()

        if "scheduled_events" not in tool_context.state:
            tool_context.state["scheduled_events"] = []
        events = list(tool_context.state["scheduled_events"])
        events.append(
            {
                "title": title,
                "date": date,
                "time": time,
                "duration_minutes": duration_minutes,
            }
        )
        tool_context.state["scheduled_events"] = events

        return (
            f"✅ Scheduled: {title}\n"
            f"📅 {date} at {time}\n"
            f"⏱️ {duration_minutes} min\n"
            f"🔔 Reminder set for 30 min before"
        )
    except Exception as e:
        return f"❌ Calendar error: {e}. Make sure credentials.json exists."


def get_upcoming_events(tool_context: ToolContext) -> str:
    """Get upcoming events from Google Calendar for the next 7 days.

    Returns: Formatted list of upcoming events
    """
    try:
        service = _get_calendar_service()
        now = datetime.now(timezone.utc)
        week_later = now + timedelta(days=7)

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=week_later.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        items = events_result.get("items", [])

        if not items:
            return "No upcoming events in the next 7 days."

        lines = ["📅 Upcoming Events (next 7 days):"]
        for ev in items:
            start = ev["start"].get("dateTime", ev["start"].get("date", ""))
            lines.append(f"  • {ev.get('summary', 'Untitled')} — {start}")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Could not fetch calendar. Make sure credentials.json exists. ({e})"


def check_focus_schedule(tool_context: ToolContext) -> str:
    """Check if there is an active or upcoming event in the next 30 minutes.

    Returns: Focus mode status
    """
    try:
        service = _get_calendar_service()
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(minutes=30)

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=window_end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        items = events_result.get("items", [])

        if not items:
            return "✅ No upcoming events. You're free to work."

        ev = items[0]
        title = ev.get("summary", "Untitled")
        start_str = ev["start"].get("dateTime", "")
        end_str = ev["end"].get("dateTime", "")

        if start_str:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str) if end_str else None

            if start_dt <= now.replace(tzinfo=start_dt.tzinfo):
                end_fmt = end_dt.strftime("%H:%M") if end_dt else "unknown"
                return f"🎯 FOCUS MODE: {title} is active until {end_fmt}. Notifications paused."
            else:
                minutes_until = int(
                    (start_dt - now.replace(tzinfo=start_dt.tzinfo)).total_seconds()
                    / 60
                )
                return f"🔔 REMINDER: {title} starts in {minutes_until} minutes. Prepare to focus."

        return "✅ No upcoming events. You're free to work."
    except Exception as e:
        # Fallback to session state
        scheduled = tool_context.state.get("scheduled_events", [])
        if scheduled:
            next_ev = scheduled[-1]
            return f"📅 (offline) Next session event: {next_ev.get('title')} on {next_ev.get('date')} at {next_ev.get('time')}. (Calendar API unavailable: {e})"
        return f"✅ No upcoming events. (Calendar API unavailable: {e})"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _start_scheduler() -> BackgroundScheduler:
    """Start background scheduler that auto-fetches jobs every N hours."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=run_auto_fetch_and_score,
        trigger="interval",
        hours=FETCH_INTERVAL_HOURS,
        id="auto_fetch_jobs",
        replace_existing=True,
        # next_run_time=datetime.utcnow(),  # uncomment for production
    )
    scheduler.start()
    print(f"⏰ Scheduler started — auto-fetching every {FETCH_INTERVAL_HOURS}h.")
    return scheduler


# ---------------------------------------------------------------------------
# Agents — all using Groq via LiteLlm
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agents — single root agent with all tools (Groq compatible)
# ---------------------------------------------------------------------------

scheduler_agent = Agent(
    name="scheduler_agent",
    model=LiteLlm(model=f"groq/{GROQ_MODEL}"),
    instruction=(
        "You manage Zohaib's Google Calendar. "
        "For scheduling requests: extract title, date, time, duration from "
        "the user's message and call schedule_event. "
        "For viewing schedule: call get_upcoming_events. "
        "For focus check: call check_focus_schedule. "
        "Always confirm the event details before scheduling. "
        "Parse natural language dates: 'tomorrow' = next day, "
        "'next Monday' = calculate the date. "
        "Return dates in YYYY-MM-DD format and times in HH:MM 24hr format."
    ),
    tools=[schedule_event, get_upcoming_events, check_focus_schedule],
)


root_agent = Agent(
    name="root_agent",
    model=LiteLlm(model=f"groq/{GROQ_MODEL}"),
    instruction=(
        "You are AgentZ, a personal concierge for Zohaib Ali — a MERN and AI "
        "developer building toward financial independence by 2030.\n\n"
        "You have these tools available:\n"
        "- score_opportunity: score a job description against Zohaib's skillset\n"
        "- draft_proposal: draft a proposal for a scored job\n"
        "- get_morning_briefing: get today's goals, finances, habits, deadlines\n"
        "- triage_emails: categorise emails as CLIENT/PAYMENT/OPPORTUNITY/IGNORE\n"
        "- auto_fetch_jobs: fetch and score jobs from Google Jobs via SerpAPI\n"
        "- get_dashboard_summary: show current dashboard job summary\n\n"
        "Sub-agents:\n"
        "- scheduler_agent: manages Google Calendar\n\n"
        "Rules:\n"
        "- Job description shared → call score_opportunity, then draft_proposal if score >= 50\n"
        "- 'morning briefing' or 'daily update' → call get_morning_briefing\n"
        "- Emails shared → call triage_emails\n"
        "- 'fetch jobs' or 'scan jobs' → call auto_fetch_jobs\n"
        "- 'dashboard' or 'show jobs' → call get_dashboard_summary\n"
        "- 'schedule', 'meeting', 'remind me', 'add to calendar', "
        "'upcoming events', \"what's on my calendar\" → scheduler_agent\n"
        "- Never send proposals automatically. Always require human approval.\n"
        "- When draft_proposal tool returns text, return the COMPLETE tool output verbatim to the user. Do not summarize or paraphrase it.\n"
    ),
    tools=[
        score_opportunity,
        draft_proposal,
        get_morning_briefing,
        triage_emails,
        auto_fetch_jobs,
        get_dashboard_summary,
    ],
    sub_agents=[scheduler_agent],
)
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_scheduler = _start_scheduler()

app = App(
    root_agent=root_agent,
    name="app",
)
