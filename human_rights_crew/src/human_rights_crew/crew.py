from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# BaseTool import differs by CrewAI version
try:
    from crewai.tools import BaseTool  # newer
except Exception:
    from crewai_tools import BaseTool  # fallback


def ensure_github_models_env() -> None:
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


def _get_mcp_base_url() -> str:
    """
    Converts MCP endpoint like:
      https://xxx.fastmcp.app/mcp
    into HTTP tool base:
      https://xxx.fastmcp.app
    """
    url = (os.getenv("MCP_URL") or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("MCP_URL is not set.")

    if url.endswith("/mcp"):
        url = url[:-4].rstrip("/")

    if not (url.startswith("http://") or url.startswith("https://")):
        raise RuntimeError(f"Invalid MCP_URL: {url}. Must start with http:// or https://")

    return url


def _call_http_tool(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    FastMCP Cloud exposes tools over HTTP typically at:
      {base}/tools/{tool}
    """
    base = _get_mcp_base_url()
    endpoint = f"{base}/tools/{tool}"

    r = requests.post(endpoint, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    if isinstance(data, dict):
        return data
    return {"result": data}


class RagAskInput(BaseModel):
    question: str = Field(..., description="User question about human rights instruments.")
    out_dir: str = Field(default="rag_out", description="Index output directory.")
    k: int = Field(default=8, description="Top-k retrieved chunks.")
    use_llm: bool = Field(default=True, description="Whether tool uses LLM.")
    provider: str = Field(default="github_models", description="Provider.")
    model: str = Field(default="openai/gpt-4o-mini", description="Model.")
    article: Optional[int] = Field(default=None, description="Optional article number.")


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
        return _call_http_tool(
            "rag_ask",
            {
                "question": question,
                "out_dir": out_dir,
                "k": k,
                "use_llm": use_llm,
                "provider": provider,
                "model": model,
                "article": article,
            },
        )


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
