from __future__ import annotations

import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


def _get_mcp_url() -> str:
    url = os.getenv("MCP_URL", "").strip()
    if not url:
        raise ValueError("MCP_URL is missing. Set it in .env or export MCP_URL in your terminal.")
    if not url.startswith("https://"):
        raise ValueError(f"Invalid MCP_URL: {url}. CrewAI requires MCP URLs to start with https://")
    return url


def _inject_mcp(cfg: Dict[str, Any]) -> Dict[str, Any]:
    new_cfg = dict(cfg)          # do not mutate original config
    new_cfg["mcps"] = [_get_mcp_url()]
    return new_cfg


@CrewBase
class HumanRightsCrew:
    """HumanRightsCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        cfg = _inject_mcp(self.agents_config["researcher"])  # type: ignore[index]
        return Agent(config=cfg, verbose=True)

    @agent
    def writer(self) -> Agent:
        cfg = _inject_mcp(self.agents_config["writer"])  # type: ignore[index]
        return Agent(config=cfg, verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])  # type: ignore[index]

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # created by @agent decorators
            tasks=self.tasks,    # created by @task decorators
            process=Process.sequential,
            verbose=True,
        )
