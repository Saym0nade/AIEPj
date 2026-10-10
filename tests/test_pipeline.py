import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from smcii import config, data, preprocess, model


def test_preprocess_transform_shape_dtype_and_reorder():
    train_df, test_df = data.split()
    preproc = preprocess.fit_save(train_df)
    X, y = data.X_y(test_df)
    X_t = preprocess.transform(X)
    assert X_t.shape[0] == len(X)
    assert X_t.dtype == np.float32

    X_reordered = X[config.ALL_FEATURES[::-1]]
    X_t2 = preprocess.transform(X_reordered)
    assert X_t2.shape == X_t.shape


def test_all_artifacts_exist():
    required = [
        config.PREPROCESSOR_PATH,
        config.FEATURE_METADATA_PATH,
        config.MODEL_TORCH_PATH,
        config.MODEL_ONNX_FP32_PATH,
        config.MODEL_ONNX_INT8_PATH,
        config.METRICS_PATH,
    ]
    for p in required:
        assert os.path.exists(p), f"Missing {p}"


def test_onnx_torch_agree():
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

    sess_fp32 = ort.InferenceSession(str(config.MODEL_ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    y_onnx_fp32 = sess_fp32.run(None, {"input": X_test_t})[0].flatten()

    assert np.allclose(y_torch, y_onnx_fp32, atol=1e-2)

    sess = ort.InferenceSession(str(config.MODEL_ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    y_onnx_int8 = sess.run(None, {"input": X_test_t})[0].flatten()
    assert np.mean(np.abs(y_torch - y_onnx_int8)) < 1.0


def test_int8_mae_delta():
    train_df, test_df = data.split()
    preproc = preprocess.load()
    X_test, y_test = data.X_y(test_df)
    X_test_t = preproc.transform(X_test).astype(np.float32)

    fp32_mae = export_onnx_eval(str(config.MODEL_ONNX_FP32_PATH), X_test_t, y_test.values)
    int8_mae = export_onnx_eval(str(config.MODEL_ONNX_INT8_PATH), X_test_t, y_test.values)
    assert abs(int8_mae - fp32_mae) < 2.0


def export_onnx_eval(path, X, y):
    from sklearn.metrics import mean_absolute_error

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y_pred = sess.run(None, {"input": X.astype(np.float32)})[0].flatten()
    return float(mean_absolute_error(y, y_pred))


def test_int8_size_budget():
    size_mb = os.path.getsize(config.MODEL_ONNX_INT8_PATH) / (1024 * 1024)
    assert size_mb < config.MODEL_SIZE_BUDGET_MB
