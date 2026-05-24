# AI Agents with Tool Use

[![CI](https://github.com/cerenaaa/ai-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/ai-agents/actions)

Tool-using LLM agent framework built on Claude's function calling API. Implements ReAct-style reasoning (Thought → Action → Observation loops), persistent memory, and composable tool registries.

## Architecture

```
User Query
    ↓
Agent (ReAct loop)
    ├── Think: plan next action
    ├── Act: call tool from registry
    ├── Observe: process tool result
    └── Repeat until answer ready → Respond
         ↑
    Tool Registry
    ├── data_analysis_tool    (run pandas/stats operations)
    ├── web_search_tool       (search + summarize)
    ├── code_executor_tool    (safe Python sandbox)
    └── memory_tool           (store/retrieve facts)
```

## Structure

```
ai-agents/
├── agents/
│   ├── base_agent.py         # ReAct loop + tool dispatch
│   └── ds_agent.py           # Data science specialist agent
├── tools/
│   ├── registry.py           # Tool registration and schema builder
│   ├── data_tools.py         # Data analysis + stats tools
│   └── code_tools.py         # Safe Python code execution
├── memory/
│   └── agent_memory.py       # Episodic + semantic memory
└── run_agent.py              # Interactive demo
```

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python run_agent.py
```

## Example

```
You: Analyze the distribution of monthly charges and flag any anomalies.

Agent: [Thinking] I need to load data, compute statistics, and check for outliers.
       [Tool: describe_dataframe] → mean=65.2, std=18.4, p99=112.0
       [Tool: detect_outliers] → 23 outliers found (z-score > 3)
       [Response] The monthly charges are roughly normally distributed (mean=$65, std=$18).
       23 accounts (0.2%) have charges exceeding $112, likely enterprise tier outliers...
```
