"""
ReAct-style agent base class.
Implements Thought → Action → Observation loops using Claude tool use.
"""
from __future__ import annotations
import json
from typing import Callable
import anthropic
from tools.registry import ToolRegistry


class ReActAgent:
    """
    ReAct agent: iteratively reasons and acts until it has an answer.
    Uses Claude's native tool_use content blocks for structured action dispatch.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = None,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        self.registry = tool_registry
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.client = anthropic.Anthropic()
        self.system = system_prompt or (
            "You are a helpful data science assistant. "
            "Use available tools to answer questions accurately. "
            "Think step by step before acting."
        )

    def _log(self, label: str, content: str):
        if self.verbose:
            print(f"  [{label}] {content[:200]}")

    def run(self, query: str, context: dict = None) -> str:
        messages = [{"role": "user", "content": query}]
        if context:
            messages[0]["content"] = f"Context: {json.dumps(context)}\n\n{query}"

        tools = self.registry.to_anthropic_tools()

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system,
                tools=tools,
                messages=messages,
            )

            # Collect text thoughts
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self._log("Thinking", block.text.strip())

            # Check stop condition
            if response.stop_reason == "end_turn":
                final = " ".join(b.text for b in response.content if b.type == "text")
                return final.strip()

            # Process tool calls
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    self._log("Action", f"{block.name}({json.dumps(block.input)[:100]})")
                    result = self.registry.call(block.name, block.input)
                    self._log("Observation", str(result)[:300])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

                messages.append({"role": "user", "content": tool_results})

        return "Max steps reached. Partial answer available from observations above."