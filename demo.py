"""Terminal demo for AgentZ core flows without running the web server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class FakeContext:
    """Minimal stand-in for ADK ToolContext."""

    def __init__(self) -> None:
        self.state: dict = {}


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def _load_skillset(skillset_path: Path) -> str:
    return skillset_path.read_text(encoding="utf-8")


def main() -> None:
    agentz_root = Path(__file__).resolve().parent

    # app.agent expects relative files like context/skillset.md and dashboard_jobs.json.
    os.chdir(agentz_root)
    sys.path.insert(0, str(agentz_root))

    try:
        from app.agent import (  # pylint: disable=import-outside-toplevel
            _load_dashboard,
            get_morning_briefing,
            score_opportunity,
            triage_emails,
        )
    except ModuleNotFoundError as exc:
        _print_header("AgentZ import error")
        print(f"Missing dependency while importing app.agent: {exc}")
        print("Install project dependencies, then rerun: demo.py")
        return

    tool_context = FakeContext()

    # 1) Load skillset.md and score a sample job description.
    _print_header("1) Skillset Load + Opportunity Score")
    skillset_path = agentz_root / "context" / "skillset.md"
    skillset_text = _load_skillset(skillset_path)
    print(f"Loaded skillset.md from: {skillset_path}")
    print(f"Skillset size: {len(skillset_text)} characters")

    sample_job_description = """
    Looking for a React and Node.js developer to build a medical AI dashboard
    with chatbot features. MongoDB backend, REST API integration, AI automation.
    Budget $2000, 6 weeks timeline.
    Scope includes React + Next.js frontend, Node.js API integration, MongoDB,
    and an AI chatbot assistant for clinic patient inquiry automation.
    """
    score_raw = score_opportunity(sample_job_description.strip(), tool_context)
    score_data = json.loads(score_raw)
    print("\nSample job score:")
    print(json.dumps(score_data, indent=2))

    # 2) Morning briefing from lmos_data.json.
    _print_header("2) Morning Briefing (from lmos_data.json)")
    briefing = get_morning_briefing(tool_context)
    print(briefing)

    # 3) Triage a sample list of emails.
    _print_header("3) Email Triage")
    sample_emails = (
        "Invoice for June services|accounts@client.com,"
        "Project revision request for clinic dashboard|pm@healthclinic.io,"
        "New gig request on Upwork|alerts@upwork.com,"
        "Weekend sale on office chairs|promo@store.com"
    )
    triage_raw = triage_emails(sample_emails)
    triaged = json.loads(triage_raw)
    for idx, item in enumerate(triaged, start=1):
        print(f"{idx}. [{item['category']}] {item['email']}")
        print(f"   Reason: {item['reason']}")

    # 4) Dashboard summary from dashboard_jobs.json.
    _print_header("4) Dashboard Summary (from dashboard_jobs.json)")
    jobs = _load_dashboard()
    total = len(jobs)
    high = sum(1 for j in jobs if j.get("score", 0) >= 70)
    mid = sum(1 for j in jobs if 50 <= j.get("score", 0) < 70)
    low = sum(1 for j in jobs if j.get("score", 0) < 50)
    by_status: dict[str, int] = {}
    for job in jobs:
        status = job.get("status", "new")
        by_status[status] = by_status.get(status, 0) + 1

    print(f"Total jobs: {total}")
    print(f"High match (>=70): {high}")
    print(f"Mid  match (50-69): {mid}")
    print(f"Low  match (<50): {low}")
    print("By status:", by_status if by_status else "{}")

    if jobs:
        print("\nTop 5 jobs:")
        for idx, job in enumerate(jobs[:5], start=1):
            print(
                f"{idx}. {job.get('title', 'Untitled')} @ {job.get('company', 'Unknown')} "
                f"(Score {job.get('score', 0)}, Status {job.get('status', 'new')})"
            )
    else:
        print("\nNo dashboard jobs found.")


if __name__ == "__main__":
    main()
