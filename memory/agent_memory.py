"""
Agent memory: episodic (conversation history) + semantic (key-value fact store).
"""
from __future__ import annotations
import json
import datetime
from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryEntry:
    key: str
    value: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    access_count: int = 0


class AgentMemory:
    def __init__(self, max_episodic: int = 50):
        self.episodic: deque[dict] = deque(maxlen=max_episodic)
        self.semantic: dict[str, MemoryEntry] = {}

    def add_episode(self, role: str, content: str):
        self.episodic.append({"role": role, "content": content,
                               "ts": datetime.datetime.utcnow().isoformat()})

    def remember(self, key: str, value: str):
        self.semantic[key.lower()] = MemoryEntry(key=key, value=value)
        return f"Stored: {key} = {value}"

    def recall(self, key: str) -> str:
        entry = self.semantic.get(key.lower())
        if not entry:
            # Fuzzy search
            matches = [k for k in self.semantic if key.lower() in k]
            if matches:
                entry = self.semantic[matches[0]]
            else:
                return f"No memory found for '{key}'"
        entry.access_count += 1
        return entry.value

    def get_context_window(self, n: int = 10) -> list[dict]:
        return [{"role": e["role"], "content": e["content"]}
                for e in list(self.episodic)[-n:]]

    def summarize(self) -> str:
        return json.dumps({
            "episodic_entries": len(self.episodic),
            "semantic_facts": len(self.semantic),
            "top_recalled": sorted(self.semantic.values(), key=lambda e: e.access_count, reverse=True)[:3]
        }, default=str)