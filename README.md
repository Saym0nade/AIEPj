# smcii — Social Media Impact on Mental Health Index

A small, reproducible ML system that turns behavioural metrics (sleep, social-media
usage, platform, device, and related lifestyle signals) into a continuous
**Mental Health Index** prediction and recommends the **minimal lifestyle change**
that would raise a person's predicted index above a healthy threshold.

The system trains a compact MLP regressor, exports it to a tiny int8 ONNX model for
low-latency local inference, and drives a **counterfactual** engine that may only vary
*actionable* features (within physical bounds) while never touching *immutable* ones.

The frozen interface spec lives in [`CONTRACT.md`](./CONTRACT.md); feature roles and
paths live in [`src/smcii/config.py`](./src/smcii/config.py).

## Architecture

```
                        data/social_media_impact.csv
                                   |
                        [ data.py: clean + 80/20 split ]
                                   |
                +------------------+------------------+
                |                                     |
     [ preprocess.py ]                       [ model.py ]
   StandardScaler(num) +                    MLP (64,32)  <-- deployed
   OneHotEncoder(cat)                       LightGBM     <-- baseline
   -> preprocessor.joblib                   -> model.pt, metrics.json
                |                                     |
                +------------------+------------------+
                                   |
                        [ export_onnx.py ]
              torch.onnx.export (fp32) -> dynamic quantize -> int8
              model_fp32.onnx        model_int8.onnx  (< 5 MB budget)
                                   |
          +------------------------+------------------------+
          |                        |                        |
  [ runtime.py ]          [ counterfactual.py ]       [ benchmark.py ]
  ONNX inference          DiCE counterfactuals         latency / memory /
  (no torch)              over ACTIONABLE only         payload -> benchmark_report.json
          |                        |
          +-----------+------------+
                      |
                [ cli.py: suggest ]
```

## Dataset & feature policy

Raw data: `data/social_media_impact.csv` — 4,500 rows x 16 columns. After cleaning
(rows with nulls in model columns dropped) 4,369 rows remain.

| Role | Features |
| --- | --- |
| Target | `Mental_Health_Index` (continuous) |
| Actionable (counterfactual may change) | `Sleep_Duration_Hours` [4, 10], `Daily_Usage_Hours` [0, 8], `Primary_Platform`, `Device_Type` |
| Immutable (never changed) | `Age`, `Gender`, `Academic_Level`, `Sleep_Quality_Score`, `Social_Comparison_Frequency`, `Perceived_Stress_Score`, `Late_Night_Usage`, `Weekend_Extra_Hours`, `Academic_Performance_GPA` |
| Dropped | `Student_ID`, `Overall_Impact` |

The preprocessor expands 13 raw features into **31** model features (7 numeric +
OneHotEncoding of 6 categorical columns). Acting only on the four actionable features
is enforced by the interface, not by convention (see `CONTRACT.md`).

## Install

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

All commands below are run from the project root with the package on `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Reproduce training & export

```bash
PYTHONPATH=src ./.venv/bin/python -m smcii.export_onnx
```

(`model.py` exposes `train_mlp` / `train_lgbm`, which persist `model.pt`,
`preprocessor.joblib`, `feature_metadata.json`, and `metrics.json`.
`export_onnx` exports fp32 and quantizes to int8, asserting the size budget.)

## CLI usage

The counterfactual CLI (runtime lane) takes a JSON profile — a dict of all
`config.ALL_FEATURES` — and a target index:

```bash
PYTHONPATH=src ./.venv/bin/python -m smcii.cli suggest --profile profile.json --target 75
# add --json for machine-readable output
```

## Testing

```bash
# lane-independent data/feature checks
PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_data_config.py -q

# end-to-end deployment verification (artifacts, ONNX parity, metrics, budgets)
PYTHONPATH=src ./.venv/bin/python scripts/verify_pipeline.py
```

## Notebook

`main.ipynb` tells the full story (EDA, feature policy, model metrics,
counterfactual demo, deployment). Execute top-to-bottom with:

```bash
./.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --allow-errors main.ipynb
```

## Results

All numbers below are read directly from `artifacts/metrics.json` and the exported
artifacts — none are hand-entered.

| Metric | Value |
| --- | --- |
| Deployed model | MLP (`deployed == "mlp"`) |
| MLP cross-validation MAE (5-fold) | **4.6725** (± 0.1610) |
| MLP holdout MAE | **4.5687** |
| MLP holdout RMSE | 5.7526 |
| LightGBM holdout MAE | 4.9667 |
| LightGBM holdout RMSE | 6.1272 |
| Train / test rows | 3,495 / 874 |
| int8 ONNX payload | **~0.0079 MB** (budget 5 MB) |

The MLP beats the LightGBM baseline on holdout MAE and is exported to a sub-10 KB int8
ONNX model, well inside the 5 MB deployment budget.

Benchmark results (latency / memory / payload versus budgets) are written by the
runtime lane to `artifacts/benchmark_report.json`:

| Metric | Value | Budget | Pass |
| --- | --- | --- | --- |
| Latency p50 / p95 / p99 | 158.16 / 170.39 / 177.66 ms | p95 ≤ 200 ms | ✅ |
| Peak RSS | 206.6 MB | ≤ 1024 MB | ✅ |
| int8 ONNX payload | 0.0079 MB | < 5 MB | ✅ |
