from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import mean_absolute_error

from smcii import config, data, model, preprocess


def export_fp32(out: str = str(config.MODEL_ONNX_FP32_PATH)) -> None:
    ckpt = torch.load(config.MODEL_TORCH_PATH, map_location="cpu")
    n_features = ckpt["n_features"]
    mlp = model.MLP(n_features)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()

    dummy = torch.randn(1, n_features, dtype=torch.float32)
    torch.onnx.export(
        mlp,
        dummy,
        out,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )


def quantize_int8(
    src: str = str(config.MODEL_ONNX_FP32_PATH),
    dst: str = str(config.MODEL_ONNX_INT8_PATH),
) -> None:
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def eval_onnx(path: str, X: np.ndarray, y: np.ndarray) -> float:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y_pred = sess.run(None, {"input": X.astype(np.float32)})[0].flatten()
    return float(mean_absolute_error(y, y_pred))


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    export_fp32()
    quantize_int8()

    train_df, test_df = data.split()
    preproc = preprocess.load()
    X_test, y_test = data.X_y(test_df)
    X_test_t = preproc.transform(X_test).astype(np.float32)

    fp32_mae = eval_onnx(str(config.MODEL_ONNX_FP32_PATH), X_test_t, y_test.values)
    int8_mae = eval_onnx(str(config.MODEL_ONNX_INT8_PATH), X_test_t, y_test.values)

    int8_size_mb = os.path.getsize(config.MODEL_ONNX_INT8_PATH) / (1024 * 1024)
    assert int8_size_mb < config.MODEL_SIZE_BUDGET_MB, f"int8 size {int8_size_mb} >= budget"

    print(f"fp32_mae={fp32_mae:.6f} int8_mae={int8_mae:.6f} int8_size_mb={int8_size_mb:.6f}")


if __name__ == "__main__":
    main()
