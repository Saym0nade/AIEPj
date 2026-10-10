"""ONNX-only inference runtime (MUST NOT import torch).

Loads the persisted sklearn preprocessor and the int8 ONNX model once and
exposes batched / single-profile prediction helpers.
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from smcii import config, preprocess

_session: Optional[ort.InferenceSession] = None
_preprocessor = None


def get_session(onnx_path=config.MODEL_ONNX_INT8_PATH) -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    return _session


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = preprocess.load()
    return _preprocessor


def predict_raw(
    df: pd.DataFrame, onnx_path=config.MODEL_ONNX_INT8_PATH
) -> np.ndarray:
    """Predict Mental_Health_Index for a DataFrame of raw features.

    Returns a 1-D float32 array of shape (n,).
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.reindex(columns=config.ALL_FEATURES)
    X = get_preprocessor().transform(df).astype(np.float32)
    session = get_session(onnx_path)
    out = session.run(None, {"input": X})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def predict_profile(profile: dict) -> float:
    """Predict for a single profile dict of raw feature values."""
    row = pd.DataFrame([profile]).reindex(columns=config.ALL_FEATURES)
    return float(predict_raw(row)[0])