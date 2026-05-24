"""
Data analysis tools for the DS agent.
Safe, stateless functions that operate on in-memory data.
"""
from __future__ import annotations
import json
import statistics
import math


def describe_list(values: list, label: str = "values") -> str:
    """Compute descriptive statistics for a numeric list."""
    if not values:
        return "Empty list"
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0
    sorted_v = sorted(values)
    p50 = sorted_v[n // 2]
    p95 = sorted_v[int(n * 0.95)]
    return json.dumps({
        "label": label, "n": n, "mean": round(mean, 3),
        "std": round(std, 3), "min": min(values), "max": max(values),
        "p50": p50, "p95": p95
    })


def detect_outliers_zscore(values: list, threshold: float = 3.0) -> str:
    """Detect outliers using Z-score method."""
    if len(values) < 3:
        return json.dumps({"outlier_count": 0, "outlier_indices": []})
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return json.dumps({"outlier_count": 0, "outlier_indices": []})
    outlier_idx = [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]
    return json.dumps({
        "outlier_count": len(outlier_idx),
        "outlier_indices": outlier_idx[:10],
        "threshold_used": threshold,
        "outlier_values": [values[i] for i in outlier_idx[:5]],
    })


def compute_correlation(x: list, y: list) -> str:
    """Compute Pearson correlation between two numeric lists."""
    if len(x) != len(y) or len(x) < 2:
        return json.dumps({"error": "Lists must be equal length and have >= 2 elements"})
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    r = num / den if den > 0 else 0
    return json.dumps({"pearson_r": round(r, 4), "n": n,
                       "interpretation": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"})


def group_summary(data: list, group_key: str, value_key: str) -> str:
    """Summarize a value by group from a list of dicts."""
    groups: dict[str, list] = {}
    for row in data:
        g = str(row.get(group_key, "unknown"))
        groups.setdefault(g, []).append(float(row.get(value_key, 0)))
    summary = {g: {"n": len(vals), "mean": round(statistics.mean(vals), 3),
                   "total": round(sum(vals), 2)} for g, vals in groups.items()}
    return json.dumps(summary)