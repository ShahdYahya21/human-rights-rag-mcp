import asyncio
import json
import os

import streamlit as st
from fastmcp import Client


# -----------------------------
# Helpers
# -----------------------------
def run_async(coro):
    """Run async code safely inside Streamlit (sync app)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


async def mcp_list_tools(mcp_url: str):
    async with Client(mcp_url) as c:
        tools = await c.list_tools()
        return tools  # return full tool objects


async def mcp_allowed_args(mcp_url: str, tool_name: str = "rag_ask"):
    """
    Reads MCP tool JSON schema to know which args are allowed.
    Prevents pydantic 'unexpected_keyword_argument' errors.
    """
    tools = await mcp_list_tools(mcp_url)
    for t in tools:
        if getattr(t, "name", None) == tool_name:
            schema = getattr(t, "inputSchema", None) or {}
            props = schema.get("properties", {}) or {}
            return set(props.keys())
    return set()


async def mcp_rag_ask(mcp_url: str, payload: dict):
    async with Client(mcp_url) as c:
        r = await c.call_tool("rag_ask", payload)

        # FastMCP returns content blocks
        if not r.content:
            return {"answer": "", "sources": []}

        text = r.content[0].text

        # Tool should return JSON; handle non-JSON gracefully
        try:
            return json.loads(text)
        except Exception:
            return {"answer": text, "sources": []}


def ensure_github_models_env():
    """
    CrewAI expects OPENAI_API_KEY even when using GitHub Models.
    If user only has GITHUB_TOKEN, map it.
    """
    if not os.getenv("OPENAI_API_KEY"):
        gh = (os.getenv("GITHUB_TOKEN") or "").strip()
        if gh:
            os.environ["OPENAI_API_KEY"] = gh

    os.environ.setdefault("OPENAI_BASE_URL", "https://models.github.ai/inference")
    os.environ.setdefault("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")


def crewai_answer(question: str) -> str:
    """
    Runs your CrewAI pipeline (researcher -> writer).
    Crew uses MCP via agents.yaml `mcps: - ${MCP_URL}`.
    """
    ensure_github_models_env()

    from human_rights_crew.crew import HumanRightsCrew

    crew = HumanRightsCrew().crew()
    result = crew.kickoff(inputs={"question": question})
    return str(result)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Human Rights RAG (MCP + CrewAI)", layout="wide")
st.title("📚 Human Rights RAG — MCP UI + CrewAI Mode")

st.sidebar.header("⚙️ Settings")

default_mcp_url = os.environ.get("MCP_URL", "https://deliberate-blue-whale.fastmcp.app/mcp")
mcp_url = st.sidebar.text_input("MCP URL", value=default_mcp_url)

out_dir = st.sidebar.text_input("RAG out_dir", value=os.environ.get("RAG_OUT_DIR", "rag_out"))

k = st.sidebar.slider("Top-k (k)", 1, 20, 8)
use_llm = st.sidebar.checkbox("Use LLM (MCP tool)", value=True)

provider = st.sidebar.selectbox("Provider (MCP tool)", ["github_models", "openai"], index=0)
model = st.sidebar.text_input("Model (MCP tool)", value="openai/gpt-4o-mini")

st.sidebar.divider()
st.sidebar.subheader("🎯 Retrieval control (MCP direct only)")
must_title = st.sidebar.text_input("must_title (hard filter)", value="")
prefer_title = st.sidebar.text_input("prefer_title (soft rerank)", value="")

st.sidebar.caption("Quick presets")
cols = st.sidebar.columns(2)
if cols[0].button("UDHR"):
    st.session_state["must_title_override"] = "Universal Declaration of Human Rights"
if cols[1].button("ICCPR"):
    st.session_state["must_title_override"] = "International Covenant on Civil and Political Rights"
if "must_title_override" in st.session_state:
    must_title = st.session_state["must_title_override"]

st.sidebar.divider()
mode = st.sidebar.radio(
    "Run mode",
    ["MCP Direct (rag_ask)", "CrewAI (researcher → writer)"],
    index=0,
)

question = st.text_area(
    "✍️ Your question",
    value="What does ICCPR say about freedom of expression?",
    height=120,
)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    btn_tools = st.button("🔍 Check tools", use_container_width=True)
with colB:
    btn_ask = st.button("🚀 Ask", use_container_width=True)
with colC:
    st.caption("Tip: MCP Direct shows chunks + scores. CrewAI gives final answer with citations.")


# -----------------------------
# Actions
# -----------------------------
if btn_tools:
    with st.spinner("Listing tools..."):
        try:
            tools = run_async(mcp_list_tools(mcp_url))
            names = [t.name for t in tools]
            st.success(f"Connected ✅ Tools: {names}")

            # Show schema for rag_ask to debug args
            for t in tools:
                if t.name == "rag_ask":
                    st.info("rag_ask input schema (allowed args):")
                    schema = getattr(t, "inputSchema", {}) or {}
                    st.json(schema.get("properties", {}))
        except Exception as e:
            st.error(f"Failed to connect to MCP: {e}")

if btn_ask:
    if mode.startswith("MCP Direct"):
        # Build full payload (from UI)
        payload_full = {
            "question": question,
            "out_dir": out_dir,
            "k": k,
            "use_llm": use_llm,
            "provider": provider,
            "model": model,
            "prefer_title": prefer_title,
            "must_title": must_title,
        }

        with st.spinner("Validating allowed args..."):
            try:
                allowed = run_async(mcp_allowed_args(mcp_url, "rag_ask"))
            except Exception as e:
                st.error(f"Could not read tool schema: {e}")
                st.stop()

        # Filter payload to only allowed keys
        payload = {kk: vv for kk, vv in payload_full.items() if kk in allowed}

        with st.spinner("Calling MCP tool rag_ask..."):
            try:
                data = run_async(mcp_rag_ask(mcp_url, payload))
            except Exception as e:
                st.error(f"MCP call failed: {e}")
                st.stop()

        answer = data.get("answer", "")
        sources = data.get("sources", []) or []

        st.subheader("✅ Answer (MCP Direct)")
        st.write(answer)

        st.subheader("🔗 Sources")
        if not sources:
            st.info("No sources returned.")
        else:
            rows = []
            for s in sources:
                rows.append({
                    "rank": s.get("rank"),
                    "score": s.get("score"),
                    "title": s.get("title"),
                    "url": s.get("url"),
                })
            st.dataframe(rows, use_container_width=True)

            st.subheader("📄 Retrieved Chunks")
            for i, s in enumerate(sources):
                title = s.get("title", "Untitled")
                url = s.get("url", "")
                score = float(s.get("score", 0.0) or 0.0)
                text = s.get("text", "")

                with st.expander(f"{i}) {title} | score={score:.3f}"):
                    if url:
                        st.markdown(f"**URL:** {url}")
                    st.write(text)

    else:
        # CrewAI mode: your agents will call MCP internally (via agents.yaml mcps)
        os.environ["MCP_URL"] = mcp_url

        with st.spinner("Running CrewAI (researcher → writer)..."):
            try:
                final_answer = crewai_answer(question)
            except Exception as e:
                st.error(f"CrewAI failed: {e}")
                st.stop()

        st.subheader("✅ Final Answer (CrewAI)")
        st.write(final_answer)
