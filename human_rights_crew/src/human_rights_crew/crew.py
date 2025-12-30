from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from fastmcp import Client
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

try:
    from crewai.tools import BaseTool  # newer CrewAI
except Exception:
    from crewai_tools import BaseTool  # fallback


def ensure_github_models_env() -> None:
    """
    CrewAI expects OPENAI_API_KEY even when using GitHub Models.
    If you only have GITHUB_TOKEN, map it to OPENAI_API_KEY.
    """
    if not os.getenv("OPENAI_API_KEY"):
        gh = (os.getenv("GITHUB_TOKEN") or "").strip()
        if gh:
            os.environ["OPENAI_API_KEY"] = gh

    os.environ.setdefault("OPENAI_BASE_URL", "https://models.github.ai/inference")
    os.environ.setdefault("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")


def _run_async(coro):
    """
    Python 3.12-safe: always create and run a fresh event loop.
    Avoids 'There is no current event loop' warnings and hanging behavior.
    """
    return asyncio.run(coro)


async def _mcp_call(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call MCP tool via MCP protocol (NOT HTTP /tools/...).
    Adds a hard timeout to avoid 'stuck' runs.
    """
    mcp_url = (os.getenv("MCP_URL") or "").strip()
    if not mcp_url:
        raise RuntimeError("MCP_URL is not set.")

    async def _do_call():
        async with Client(mcp_url) as c:
            r = await c.call_tool(tool, payload)
            if not getattr(r, "content", None):
                return {"answer": "", "sources": []}

            text = r.content[0].text
            try:
                return json.loads(text)
            except Exception:
                return {"answer": text, "sources": []}

    # ⏱️ hard timeout (seconds)
    return await asyncio.wait_for(_do_call(), timeout=45)


class RagAskInput(BaseModel):
    question: str = Field(..., description="User question about human rights instruments.")
    out_dir: str = Field(default="rag_out", description="Index output directory.")
    k: int = Field(default=8, description="Top-k retrieved chunks.")
    use_llm: bool = Field(default=True, description="Whether tool uses LLM to format answer.")
    provider: str = Field(default="github_models", description="Provider used by MCP tool.")
    model: str = Field(default="openai/gpt-4o-mini", description="Model used by MCP tool.")
    article: Optional[int] = Field(default=None, description="Optional article number (e.g., 19).")


class RagAskTool(BaseTool):
    name: str = "rag_ask"
    description: str = "Call MCP Human Rights RAG tool to retrieve answer + sources."
    args_schema: type[BaseModel] = RagAskInput

    def _run(
        self,
        question: str,
        out_dir: str = "rag_out",
        k: int = 8,
        use_llm: bool = True,
        provider: str = "github_models",
        model: str = "openai/gpt-4o-mini",
        article: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "question": question,
            "out_dir": out_dir,
            "k": k,
            "use_llm": use_llm,
            "provider": provider,
            "model": model,
            "article": article,
        }
        return _run_async(_mcp_call("rag_ask", payload))


@CrewBase
class HumanRightsCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        cfg = self.agents_config["researcher"]
        return Agent(
            role=cfg.get("role"),
            goal=cfg.get("goal"),
            backstory=cfg.get("backstory"),
            verbose=bool(cfg.get("verbose", True)),
            tools=[RagAskTool()],
        )

    @agent
    def writer(self) -> Agent:
        cfg = self.agents_config["writer"]
        return Agent(
            role=cfg.get("role"),
            goal=cfg.get("goal"),
            backstory=cfg.get("backstory"),
            verbose=bool(cfg.get("verbose", True)),
            # No tools here: writer only uses evidence from context
        )

    @task
    def find_evidence(self) -> Task:
        cfg = self.tasks_config["find_evidence"]
        return Task(
            description=cfg.get("description"),
            expected_output=cfg.get("expected_output"),
            agent=self.researcher(),
        )

    @task
    def write_answer(self) -> Task:
        cfg = self.tasks_config["write_answer"]
        return Task(
            description=cfg.get("description"),
            expected_output=cfg.get("expected_output"),
            agent=self.writer(),
            context=[self.find_evidence()],
        )

    @crew
    def crew(self) -> Crew:
        ensure_github_models_env()
        return Crew(
            agents=[self.researcher(), self.writer()],
            tasks=[self.find_evidence(), self.write_answer()],
            process=Process.sequential,
            verbose=True,
        )
