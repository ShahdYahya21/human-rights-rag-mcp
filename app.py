import asyncio
import json
import streamlit as st
from fastmcp import Client

MCP_URL = "http://127.0.0.1:8000/mcp"


def run_async(coro):
    return asyncio.run(coro)


async def list_tools_async():
    async with Client(MCP_URL) as client:
        return await client.list_tools()


async def call_tool_async(name: str, args: dict):
    async with Client(MCP_URL) as client:
        res = await client.call_tool(name, args)
        # FastMCP returns structured content; try common access patterns
        if hasattr(res, "data"):
            return res.data
        return res


st.set_page_config(page_title="MCP Tool UI", layout="wide")
st.title("🔧 Human Rights MCP Tool UI")
st.caption(f"MCP endpoint: {MCP_URL}")

if st.button("🔄 Refresh MCP Tools"):
    st.session_state["tools"] = run_async(list_tools_async())

tools = st.session_state.get("tools")
if not tools:
    st.info("Click **Refresh MCP Tools** to load tools from the MCP server.")
    st.stop()

tool_names = [t.name for t in tools]
st.subheader("✅ Available MCP Tools (Obvious)")
st.write(tool_names)

selected = st.selectbox("Select MCP Tool", tool_names)

st.subheader("Tool Inputs")
args = {}

if selected == "rag_answer":
    q = st.text_area("Question", "What does the CRPD say about accessibility?")
    k = st.slider("Top-k", 1, 10, 5)
    model = st.text_input("Model", "openai/gpt-4o-mini")
    args = {"question": q, "k": int(k), "model": model, "provider": "github_models", "out_dir": "rag_out"}

elif selected == "serper_search":
    query = st.text_input("Search query", "Universal Declaration of Human Rights freedom of expression Article 19")
    n = st.slider("n_results", 1, 10, 5)
    args = {"query": query, "n_results": int(n)}

elif selected == "calendar_create_event":
    title = st.text_input("Title", "Read CRPD Article 9")
    start_iso = st.text_input("Start ISO", "2025-12-27T15:00:00+02:00")
    end_iso = st.text_input("End ISO", "2025-12-27T15:30:00+02:00")
    desc = st.text_area("Description", "Study accessibility obligations and write notes.")
    args = {"title": title, "start_iso": start_iso, "end_iso": end_iso, "description": desc}

else:
    st.warning("No UI defined for this tool yet. Provide JSON args manually.")
    raw = st.text_area("Args (JSON)", "{}")
    args = json.loads(raw)

if st.button("▶ Run MCP Tool", use_container_width=True):
    out = run_async(call_tool_async(selected, args))
    st.subheader("Tool Output")
    st.json(out)
