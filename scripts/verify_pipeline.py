"""Standalone deployment-spec verification harness.

Run with:
    PYTHONPATH=src ./.venv/bin/python scripts/verify_pipeline.py

Prints a PASS/FAIL line per check and a summary. Exit code 0 = all pass.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import mean_absolute_error

from smcii import config, data, model, preprocess

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    print("=" * 70)
    print("smcii deployment verification")
    print("=" * 70)

    # 1. required artifacts exist
    required = {
        "preprocessor.joblib": config.PREPROCESSOR_PATH,
        "feature_metadata.json": config.FEATURE_METADATA_PATH,
        "model.pt": config.MODEL_TORCH_PATH,
        "model_fp32.onnx": config.MODEL_ONNX_FP32_PATH,
        "model_int8.onnx": config.MODEL_ONNX_INT8_PATH,
        "metrics.json": config.METRICS_PATH,
    }
    missing = [name for name, p in required.items() if not os.path.exists(p)]
    check(
        "all required artifacts exist",
        not missing,
        "ok" if not missing else f"missing: {', '.join(missing)}",
    )
    if missing:
        return _summary()

    # 2. int8 payload budget
    int8_size_mb = os.path.getsize(config.MODEL_ONNX_INT8_PATH) / (1024 * 1024)
    check(
        "int8 model size < budget",
        int8_size_mb < config.MODEL_SIZE_BUDGET_MB,
        f"{int8_size_mb:.4f} MB < {config.MODEL_SIZE_BUDGET_MB} MB",
    )

    # 3/4. ONNX vs torch agreement + MAE
    train_df, test_df = data.split()
    preproc = preprocess.load()
    X_test, y_test = data.X_y(test_df)
    X_test_t = preproc.transform(X_test).astype(np.float32)

    ckpt = torch.load(config.MODEL_TORCH_PATH, map_location="cpu")
    mlp = model.MLP(ckpt["n_features"])
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()
    with torch.no_grad():
        y_torch = mlp(torch.from_numpy(X_test_t)).numpy().flatten()

    def _onnx_preds(path: str) -> np.ndarray:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess.run(None, {"input": X_test_t})[0].flatten()

    y_fp32 = _onnx_preds(str(config.MODEL_ONNX_FP32_PATH))
    y_int8 = _onnx_preds(str(config.MODEL_ONNX_INT8_PATH))

    fp32_diff = float(np.mean(np.abs(y_torch - y_fp32)))
    int8_diff = float(np.mean(np.abs(y_torch - y_int8)))
    check(
        "fp32 ONNX matches torch",
        np.allclose(y_torch, y_fp32, atol=1e-2),
        f"mean abs diff {fp32_diff:.6f}",
    )
    check(
        "int8 ONNX close to torch (< 1.0)",
        int8_diff < 1.0,
        f"mean abs diff {int8_diff:.6f}",
    )

    fp32_mae = float(mean_absolute_error(y_test.values, y_fp32))
    int8_mae = float(mean_absolute_error(y_test.values, y_int8))
    check(
        "int8 holdout MAE finite and close to fp32",
        np.isfinite(int8_mae) and abs(int8_mae - fp32_mae) < 2.0,
        f"fp32 {fp32_mae:.4f} vs int8 {int8_mae:.4f} (Δ {abs(int8_mae - fp32_mae):.4f})",
    )

    # 5. metrics.json contents
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

    mlp_metrics = metrics.get("mlp", {})
    check(
        "metrics: MLP cv_mae + holdout_mae present",
        "cv_mae" in mlp_metrics and "holdout_mae" in mlp_metrics,
        f"cv_mae={mlp_metrics.get('cv_mae')}, holdout_mae={mlp_metrics.get('holdout_mae')}",
    )
    check(
        "metrics: lightgbm present",
        isinstance(metrics.get("lightgbm"), dict) and bool(metrics["lightgbm"]),
        f"holdout_mae={metrics.get('lightgbm', {}).get('holdout_mae')}",
    )
    check(
        'metrics: deployed == "mlp"',
        metrics.get("deployed") == "mlp",
        f"deployed={metrics.get('deployed')!r}",
    )

    return _summary()


def _summary() -> int:
    n_total = len(RESULTS)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = n_total - n_pass
    print("-" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} passed, {n_fail} failed")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
