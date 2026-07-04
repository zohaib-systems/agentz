# AgentZ — Personal Freelance & Life Concierge Agent

**Track:** Concierge Agents
**Built by:** Zohaib Ali — MERN & AI Developer ([zohaib-systems.tech](https://zohaib-systems.tech))
**Submission for:** AI Agents Intensive Vibe Coding Capstone (Kaggle x Google)

---

## The Problem

I'm a freelance developer and microbiology undergraduate juggling client work, job hunting, and a personal Life Management OS — across Upwork, Fiverr, Google Jobs, Gmail, and Google Calendar, with zero connection between any of them.

Every day looked like this: manually scanning job boards for relevant work, rewriting the same proposal pitch from scratch, missing client emails buried in newsletters, and losing track of study sessions against my own goals.

I didn't need another dashboard to read. I needed an agent that already knows who I am — and acts on my behalf.

That's AgentZ.

---

## What AgentZ Does

AgentZ is a multi-agent personal concierge that runs in the background and surfaces only what matters: scored job opportunities, drafted proposals in my own tone, triaged emails, scheduled meetings with automatic reminders, and a morning briefing — all grounded in two files that define who I am as a developer and what I'm working toward.

It also runs a Focus Mode with Pomodoro tracking for deep work sessions, so the same agent that finds me work also protects the time I spend doing it.

---

## Why Agents (Not Just a Script)

A simple script can fetch job listings. It can't decide whether a job is worth pursuing, draft a proposal that sounds like me, hold a conversation about my calendar, or pause and wait for my approval before anything goes out the door.

Agents are the right fit here because the problem is fundamentally about judgment under context — matching opportunities against a personal skillset, reasoning about scheduling conflicts in natural language, and making decisions that still require a human in the loop before anything client-facing happens. Each sub-agent specializes in one judgment call; the orchestrator routes intent without me ever needing to know which agent is doing the work.

---

## Architecture

```
                         ┌─────────────────────────┐
                         │   AgentZ Dashboard       │
                         │   (React + Tailwind)     │
                         │  Opportunities · Chat ·  │
                         │  Email · Briefing ·      │
                         │  Proposals · Focus       │
                         └────────────┬─────────────┘
                                      │ REST (ADK /run)
                         ┌────────────▼─────────────┐
                         │       root_agent          │
                         │   (ADK 2.0 Orchestrator)  │
                         └────────────┬─────────────┘
                                      │
        ┌──────────────┬─────────────┼─────────────┬──────────────┐
        ▼              ▼             ▼             ▼              ▼
 opportunity_agent proposal_agent life_sync_agent email_triage_agent scheduler_agent
        │              │             │             │              │
        ▼              ▼             ▼             ▼              ▼
 score_opportunity draft_proposal get_morning_  triage_emails  schedule_event
  (fuzzy match)    (Groq LLM)     briefing                    get_upcoming_events
        │                          (LM-OS data)                check_focus_schedule
        ▼                                                       (Google Calendar API)
 ┌──────────────┐
 │ fetch_agent   │ ── auto_fetch_jobs (SerpAPI + APScheduler, every 4h)
 └──────────────┘     → dashboard_jobs.json

        Context Layer (the agent's memory):
        context/skillset.md     → stack, deliverables, proposal tone, keywords
        context/lmos_data.json  → goals, finances, habits, deadlines (from my LM-OS)
```

Every agent decision traces back to `skillset.md` and `lmos_data.json`. Update either file and the entire system adapts — no prompt engineering required.

---

## Course Concepts Demonstrated

### 1. Multi-Agent System (ADK 2.0)
Built with Google's Agent Development Kit using the graph Workflow API. A `root_agent` orchestrator routes user intent to specialized sub-agents — `opportunity_agent`, `proposal_agent`, `life_sync_agent`, `email_triage_agent`, `fetch_agent`, and `scheduler_agent` — each scoped to only the tools it needs. See `app/agent.py`.

### 2. Security Features
Every tool that processes external text (job descriptions, scheduling requests) runs through `security_screen()` first, which detects prompt injection attempts (`"ignore previous instructions"`, `"jailbreak"`, etc.) and blocks them before any LLM call is made. Proposals never send automatically — every draft requires explicit human approval (`status: "pending_review"` → `"approved"`), which is a deliberate human-in-the-loop design choice that also keeps AgentZ compliant with Upwork and Fiverr's prohibition on automated proposal submission.

### 3. Agent Skills (Agents CLI)
The entire project was scaffolded, linted, and evaluated using `agents-cli`: `agents-cli create` for scaffolding, `agents-cli lint` for code quality (ruff, codespell, type checking via `ty`), and `agents-cli eval generate` / `eval grade` against a custom dataset in `tests/eval/` with metrics for response quality, security compliance, and correct tool routing.

### 4. Deployability
AgentZ runs locally via `adk web` and is deployable to Google Cloud Agent Runtime using the same `agents-cli deploy` workflow scaffolded into the project (see `deployment/terraform/`). The dashboard is a static single-file app that can be served from any static host or Cloud Run.

### 5. Antigravity
The agent backend, Calendar integration, and Focus Mode UI were built using Antigravity as the vibe-coding interface — describing functionality in natural language and reviewing/iterating on the generated ADK code, rather than hand-writing every line.

---

## What's Inside

| File | Purpose |
|---|---|
| `app/agent.py` | All agents, tools, and orchestration logic |
| `context/skillset.md` | Agent's knowledge of my stack, deliverables, and proposal tone |
| `context/lmos_data.json` | My goals, finances, habits, and deadlines (LM-OS export) |
| `dashboard/index.html` | Single-file React dashboard (no build step) |
| `tests/eval/datasets/basic-dataset.json` | Evaluation test cases |
| `tests/eval/eval_config.yaml` | LLM-as-judge metrics for grading agent responses |

---

## Setup

### 1. Clone and install
```bash
git clone https://github.com/zohaib-systems/agentz
cd agentz
uv sync
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your own keys:
```
GROQ_API_KEY=your_groq_key_here
SERPAPI_KEY=your_serpapi_key_here
FETCH_INTERVAL_HOURS=4
```

Get a free Groq key at [console.groq.com](https://console.groq.com) and a free SerpAPI key at [serpapi.com](https://serpapi.com) (100 free searches/month).

### 3. (Optional) Google Calendar integration
To enable `scheduler_agent`, create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com) for the Calendar API, download as `credentials.json`, and place it in the project root. On first use, AgentZ will open a browser window for one-time authorization.

### 4. Personalize the agent
Replace `context/skillset.md` with your own skills, deliverables, and tone — and `context/lmos_data.json` with your own goals and habits. AgentZ adapts automatically.

### 5. Run
```bash
uv run adk web
```
Open `http://127.0.0.1:8000` for the ADK developer UI, or serve `dashboard/index.html` separately:
```bash
python -m http.server 5500
```
Open `http://localhost:5500/dashboard/index.html`.

---

## Example Interactions

**Job scoring:**
```
Input:  "React developer for medical AI dashboard, Node.js, MongoDB. $2000, 6 weeks."
Output: { "score": 78, "matched_skills": ["React developer", "Node.js", "AI chatbot", "medical app"], "reasoning": "Strong match" }
→ Automatically triggers proposal_agent
```

**Morning briefing:**
```
🌅 Good morning, Zohaib!
🎯 Top Goal: Financial Independence (Deadline: 2030-12-31)
💰 Monthly Surplus: 213,333 PKR
💪 Active Habit: Exercise (Day 1 of 21)
💼 New Jobs in Dashboard: 12 (3 high-quality ≥70)
```

**Email triage:**
```
Input:  "New order received|order@fiverr.com, Payment of $150|payments@paypal.com"
Output: OPPORTUNITY → New order received
        PAYMENT     → Payment of $150 received
```

---

## Design Decisions Worth Noting

**No auto-submission of proposals.** Both Upwork and Fiverr explicitly prohibit automated proposal submission. AgentZ drafts and queues every proposal for manual copy-paste — this isn't a limitation, it's the correct design for a tool that respects platform terms of service while still saving the time-consuming part of the work.

**Fuzzy matching over exact keywords.** `score_opportunity` uses RapidFuzz's `partial_ratio` and `token_set_ratio` instead of exact string matching, so a job mentioning "Node" still matches a skillset listing "Node.js" — handling the natural variation in how job posts are written.

**File-based context instead of a database.** `skillset.md` and `lmos_data.json` are plain files specifically so the agent's "personality" can be forked and replaced by anyone — no schema migration, no database setup, just edit a markdown file.

---

## What's Next

- WhatsApp integration for mobile alerts when high-scoring jobs arrive
- Multi-user support — any freelancer can deploy AgentZ with their own `skillset.md`
- MCP server exposing LM-OS data as a standardized tool interface for other agents
- Fiverr webhook integration for real-time order notifications

AgentZ isn't a capstone project I'll archive after submission — it's the infrastructure I'm building my freelance career on.

---

## Tech Stack

ADK 2.0 (graph Workflow API) · Groq (Llama 4 Scout via LiteLLM) · SerpAPI · Google Calendar API · APScheduler · RapidFuzz · React (CDN, no build step) · agents-cli, Gemini API

---

*Built during the Kaggle x Google 5-Day AI Agents Intensive, June 2026.*
