import os
import requests
from crewai.tools import tool

@tool("rag_ask")
def rag_ask(question: str, k: int = 8) -> dict:
    mcp_url = (os.getenv("MCP_URL") or "").rstrip("/")
    if not mcp_url:
        raise RuntimeError("MCP_URL is not set")

    # Adjust if your FastMCP endpoint differs:
    url = f"{mcp_url}/tools/rag_ask"

    payload = {"question": question, "k": k}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()
