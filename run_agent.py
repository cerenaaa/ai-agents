"""
Interactive DS agent demo.
Usage: ANTHROPIC_API_KEY=your_key python run_agent.py
"""
import json
from agents.base_agent import ReActAgent
from tools.registry import ToolRegistry
from tools.data_tools import describe_list, detect_outliers_zscore, compute_correlation
from memory.agent_memory import AgentMemory

import numpy as np


def build_agent() -> ReActAgent:
    registry = ToolRegistry()
    registry.register(describe_list, name="describe_data",
                      description="Compute descriptive statistics for a numeric list")
    registry.register(detect_outliers_zscore, name="detect_outliers",
                      description="Detect outliers in a numeric list using Z-score")
    registry.register(compute_correlation, name="correlation",
                      description="Compute Pearson correlation between two numeric lists")
    return ReActAgent(registry, verbose=True)


def main():
    rng = np.random.default_rng(42)
    sample_charges = list(rng.normal(65, 18, 200).clip(10, 200).round(2))
    sample_charges.extend([250.0, 300.0, 275.0])  # add outliers

    agent = build_agent()

    queries = [
        "Describe the distribution of these monthly charges and identify any outliers.",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        context = {"monthly_charges": sample_charges}
        answer = agent.run(q, context=context)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()