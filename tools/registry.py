"""
Tool registry: register Python functions as LLM-callable tools.
Automatically builds Anthropic tool schemas from type hints and docstrings.
"""
from __future__ import annotations
import inspect
import re
from typing import Callable, get_type_hints


PYTHON_TO_JSON_TYPES = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[Callable, dict]] = {}

    def register(self, func: Callable = None, *, name: str = None, description: str = None):
        """Decorator to register a function as a tool."""
        def decorator(f):
            tool_name = name or f.__name__
            tool_desc = description or (f.__doc__ or "").strip().split("\n")[0]
            schema = self._build_schema(f)
            self._tools[tool_name] = (f, {
                "name": tool_name,
                "description": tool_desc,
                "input_schema": schema,
            })
            return f
        return decorator(func) if func else decorator

    def _build_schema(self, func: Callable) -> dict:
        sig = inspect.signature(func)
        hints = {}
        try:
            hints = get_type_hints(func)
        except Exception:
            pass
        doc = inspect.getdoc(func) or ""

        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            type_name = hints.get(param_name, str).__name__ if hasattr(hints.get(param_name, str), "__name__") else "string"
            json_type = PYTHON_TO_JSON_TYPES.get(type_name, "string")
            # Extract param description from docstring
            match = re.search(rf"{param_name}:\s*(.+)", doc)
            param_desc = match.group(1).strip() if match else f"The {param_name} parameter"
            properties[param_name] = {"type": json_type, "description": param_desc}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def call(self, name: str, inputs: dict):
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        func, _ = self._tools[name]
        try:
            return func(**inputs)
        except Exception as e:
            return f"Tool error: {e}"

    def to_anthropic_tools(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]