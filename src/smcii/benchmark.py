"""Benchmark the runtime lane: latency, memory, and payload size.

Run with:
    PYTHONPATH=src ./.venv/bin/python -m smcii.benchmark
"""

from __future__ import annotations

import json
import os
import resource
import time
from typing import Dict

import numpy as np

from smcii import config, counterfactual, data, runtime

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _pick_below_threshold_profile() -> dict:
    _, test_df = data.split()
    below = test_df[test_df[config.TARGET] < config.DEFAULT_TARGET_THRESHOLD]
    if len(below) == 0:
        below = test_df
    return below[config.ALL_FEATURES].iloc[0].to_dict()


def latency(n: int = 100, warmup: int = 5) -> Dict[str, float]:
    """Time one ``counterfactual.suggest`` call (fallback: predict_raw)."""
    profile = _pick_below_threshold_profile()
    pred = runtime.predict_profile(profile)
    target = pred + 8.0

    def one_call():
        return counterfactual.suggest(profile, target=target)

    for _ in range(warmup):
        one_call()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        one_call()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(times.mean()),
    }


def memory_peak_mb() -> Dict[str, float]:
    """Peak RSS of this process (ru_maxrss, KB->MB) plus current RSS via psutil."""
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / 1024.0
    current_mb = None
    if psutil is not None:
        current_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    return {"peak_rss_mb": float(peak_mb), "current_rss_mb": float(current_mb)}


def payload_sizes() -> Dict[str, float]:
    return {
        "onnx_int8_mb": os.path.getsize(config.MODEL_ONNX_INT8_PATH) / (1024.0 * 1024.0),
        "preprocessor_kb": os.path.getsize(config.PREPROCESSOR_PATH) / 1024.0,
    }


def run() -> dict:
    lat = latency()
    mem = memory_peak_mb()
    payload = payload_sizes()

    report = {
        "latency_ms": lat,
        "memory_mb": mem,
        "payload": payload,
        "budgets": {
            "latency_budget_ms": config.LATENCY_BUDGET_MS,
            "memory_budget_mb": config.MEMORY_BUDGET_MB,
            "model_size_budget_mb": config.MODEL_SIZE_BUDGET_MB,
        },
        "pass": {
            "latency_p95": lat["p95_ms"] <= config.LATENCY_BUDGET_MS,
            "memory_peak": mem["peak_rss_mb"] <= config.MEMORY_BUDGET_MB,
            "model_size": payload["onnx_int8_mb"] <= config.MODEL_SIZE_BUDGET_MB,
        },
    }

    config.BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BENCHMARK_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=== smcii benchmark ===")
    print(f"latency  p50={lat['p50_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms  "
          f"p99={lat['p99_ms']:.2f}ms  mean={lat['mean_ms']:.2f}ms")
    print(f"memory   peak={mem['peak_rss_mb']:.1f}MB  current={mem['current_rss_mb']:.1f}MB")
    print(f"payload  int8={payload['onnx_int8_mb']:.4f}MB  preprocessor={payload['preprocessor_kb']:.1f}KB")
    print(f"pass     latency_p95={report['pass']['latency_p95']}  "
          f"memory_peak={report['pass']['memory_peak']}  model_size={report['pass']['model_size']}")
    print(f"report -> {config.BENCHMARK_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()