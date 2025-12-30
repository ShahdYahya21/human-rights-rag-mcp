from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastmcp import FastMCP

# Silence HF tokenizers fork warning (optional)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

mcp = FastMCP("HumanRightsAgentMCP")

DEFAULT_OUT_DIR = "rag_out"


# ---------------------------
# MCP TOOL 1: RAG Answer
# ---------------------------
@mcp.tool
def rag_answer(
    question: str,
    k: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
    provider: str = "github_models",
    model: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Answer using local RAG index + LLM, return answer + sources.
    """
    try:
        from human_rights_rag import answer_question
    except Exception as e:
        return {"error": "Could not import answer_question from human_rights_rag.py", "details": repr(e)}

    try:
        answer, retrieved = answer_question(
            question=question,
            k=k,
            out_dir=out_dir,
            use_llm=True,
            provider=provider,
            model=model,
        )
    except Exception as e:
        return {"error": "RAG failed", "details": repr(e)}

    sources = []
    for r in (retrieved or []):
        sources.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
            }
        )

    return {"answer": answer, "sources": sources}


# ---------------------------
# MCP TOOL 2: Serper Search
# ---------------------------
@mcp.tool
def serper_search(query: str, n_results: int = 5, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Online search using Serper.dev.
    Endpoint default is https://google.serper.dev/search. :contentReference[oaicite:1]{index=1}
    """
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return {"error": "SERPER_API_KEY is not set"}

    url = "https://google.serper.dev/search"  # default Serper endpoint :contentReference[oaicite:2]{index=2}
    payload = {"q": query, "num": n_results}
    if country:
        payload["gl"] = country  # Serper supports country/location params (gl)

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": "Serper request failed", "details": repr(e)}

    # Keep only useful fields
    organic = data.get("organic", [])[:n_results]
    results = []
    for r in organic:
        results.append(
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
            }
        )
    return {"query": query, "results": results}


# ---------------------------
# MCP TOOL 3: Google Calendar Create Event
# ---------------------------
def _get_calendar_service():
    """
    Uses Google Calendar API Python quickstart style OAuth flow. :contentReference[oaicite:3]{index=3}
    Requires:
      - credentials.json in project root
      - token.json will be created after first auth
    """
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError("credentials.json not found. Download OAuth client secrets from Google Cloud.")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


@mcp.tool
def calendar_create_event(
    title: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    """
    Create a Google Calendar event.
    start_iso/end_iso example: 2025-12-27T15:00:00+02:00
    """
    try:
        service = _get_calendar_service()
        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
        }
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        return {
            "status": "created",
            "htmlLink": created.get("htmlLink"),
            "id": created.get("id"),
        }
    except Exception as e:
        return {"error": "Calendar create failed", "details": repr(e)}


if __name__ == "__main__":
    # HTTP server endpoint becomes: http://localhost:8000/mcp :contentReference[oaicite:4]{index=4}
    mcp.run(transport="http", host="127.0.0.1", port=8000)
