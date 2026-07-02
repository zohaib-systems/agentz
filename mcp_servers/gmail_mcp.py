"""Local Gmail MCP server.

Exposes 3 tools to the AgentZ email_mcp_agent via SSE on http://localhost:8001/sse.
Run with:  uv run python mcp_servers/gmail_mcp.py

OAuth credentials / token.pickle are shared with the main agent (same project).
"""

import base64
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths — works whether run from project root or mcp_servers/
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent  # …/agentz/mcp_servers
_ROOT = _HERE.parent  # …/agentz

CALENDAR_CREDS = str(_ROOT / "credentials.json")
CALENDAR_TOKEN = str(_ROOT / "token.pickle")

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# ---------------------------------------------------------------------------
# OAuth helper (mirrors _get_calendar_service pattern)
# ---------------------------------------------------------------------------


def _get_gmail_service():
    """Return an authenticated Gmail API service resource."""
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if Path(CALENDAR_TOKEN).exists():
        with open(CALENDAR_TOKEN, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CALENDAR_CREDS, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(CALENDAR_TOKEN, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(headers: list[dict], name: str) -> str:
    """Extract a single header value from a Gmail message headers list."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _message_to_dict(msg: dict) -> dict:
    """Convert a Gmail message resource to a flat summary dict."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "id": msg.get("id", ""),
        "thread_id": msg.get("threadId", ""),
        "subject": _header(headers, "Subject"),
        "sender": _header(headers, "From"),
        "date": _header(headers, "Date"),
        "snippet": msg.get("snippet", ""),
    }


def _decode_body_part(part: dict) -> str:
    """Decode a single MIME body part (text/plain preferred)."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(payload: dict) -> str:
    """Recursively extract plain text from a MIME payload."""
    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        return _decode_body_part(payload)

    if mime_type == "text/html" and not parts:
        # Fallback: return raw html if no plain-text alternative exists
        return _decode_body_part(payload)

    text_parts = []
    for part in parts:
        text_parts.append(_extract_text(part))
    return "\n".join(filter(None, text_parts))


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("gmail-mcp", host="localhost", port=8001)


@mcp.tool()
def search_emails(query: str) -> list[dict]:
    """Search Gmail messages matching *query* (same syntax as the Gmail search bar).

    Returns up to 10 results, each with: id, thread_id, subject, sender, date, snippet.
    """
    service = _get_gmail_service()
    result = (
        service.users().messages().list(userId="me", q=query, maxResults=10).execute()
    )
    messages = result.get("messages", [])
    output = []
    for m in messages:
        full = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            )
            .execute()
        )
        output.append(_message_to_dict(full))
    return output


@mcp.tool()
def get_unread_emails() -> list[dict]:
    """Return all unread messages in the inbox.

    Returns up to 10 results, each with: id, thread_id, subject, sender, date, snippet.
    """
    return search_emails("is:unread in:inbox")


@mcp.tool()
def get_email_thread(thread_id: str) -> dict:
    """Return the full content of a Gmail thread.

    Args:
        thread_id: The thread ID (available in search_emails / get_unread_emails results).

    Returns a dict with: thread_id, message_count, messages (list of
    {id, subject, sender, date, body}).
    """
    service = _get_gmail_service()
    thread = service.users().threads().get(userId="me", id=thread_id).execute()
    msgs = thread.get("messages", [])
    output_messages = []
    for msg in msgs:
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        output_messages.append(
            {
                "id": msg.get("id", ""),
                "subject": _header(headers, "Subject"),
                "sender": _header(headers, "From"),
                "date": _header(headers, "Date"),
                "body": _extract_text(payload),
            }
        )
    return {
        "thread_id": thread_id,
        "message_count": len(output_messages),
        "messages": output_messages,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[Gmail MCP] Starting SSE server on http://localhost:8001/sse …")
    mcp.run(transport="sse")
