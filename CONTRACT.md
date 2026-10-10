# smcii — Frozen Interface Contract

Project root: `/home/sayan/Desktop/AI Essentials/Project`
Run code with: `PYTHONPATH=src ./.venv/bin/python ...` (package name: `smcii`).
All data/artifacts stay under the project root: `data/`, `artifacts/`, `src/`.

## Feature roles (from src/smcii/config.py — DO NOT EDIT config after bootstrap)
- Actionable (counterfactual engine MAY change): `Sleep_Duration_Hours` [4,10], `Daily_Usage_Hours` [0,8], `Primary_Platform` (cat), `Device_Type` (cat).
- Immutable (counterfactual MUST NOT change): `Age`, `Gender`, `Academic_Level`, `Sleep_Quality_Score`, `Social_Comparison_Frequency`, `Perceived_Stress_Score`, `Late_Night_Usage`, `Weekend_Extra_Hours`, `Academic_Performance_GPA`.
- Dropped: `Student_ID`, `Overall_Impact`.
- Target: `Mental_Health_Index` (continuous regression).

## Module interfaces (owners in parentheses)
### data.py (pipeline)
- `load_raw() -> pd.DataFrame` reads `config.RAW_DATASET`.
- `clean(df) -> pd.DataFrame` keeps `config.ALL_FEATURES + [config.TARGET]`, coerces dtypes, drops rows with nulls in those columns.
- `split(df=None, test_size=config.TEST_SIZE, seed=config.RANDOM_SEED) -> (train_df, test_df)` (loads+cleans if df None).
- `X_y(df) -> (pd.DataFrame[ALL_FEATURES], pd.Series[TARGET])`.

### preprocess.py (pipeline)
- `build() -> ColumnTransformer`: `StandardScaler(config.NUMERIC_FEATURES)` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)(config.CATEGORICAL_FEATURES)`.
- `fit_save(train_df) -> ColumnTransformer` fits on `data.X_y(train_df)[0]`, `joblib.dump` to `config.PREPROCESSOR_PATH`.
- `load() -> ColumnTransformer` from `config.PREPROCESSOR_PATH`.
- `transform(df) -> np.ndarray float32`: accepts a DataFrame containing raw `config.ALL_FEATURES` columns; returns transformed matrix.
- Also write `config.FEATURE_METADATA_PATH` JSON: `{"numeric": [...], "categorical": [...], "ohe_categories": {col: [...]}, "n_features_in": int, "n_features_out": int, "actionable": [...], "immutable": [...], "permitted_range": {...}}`.

### model.py (pipeline)
- `class MLP(nn.Module)`: `Linear(n,64)-ReLU-Linear(64,32)-ReLU-Linear(32,1)`.
- `train_mlp(train_df) -> (model, metrics_dict)`: 5-fold CV MAE (`config.CV_FOLDS`) on standardized input via preprocess; target predicted in ORIGINAL units (de-normalization folded into the final linear layer); refit final model on all train; `torch.save({"state_dict":..., "n_features":...}, config.MODEL_TORCH_PATH)`.
- `train_lgbm(train_df) -> (model, metrics_dict)`: `lightgbm.LGBMRegressor` baseline.
- Write `config.METRICS_PATH`: `{"mlp":{"cv_mae":..,"holdout_mae":..,"rmse":..},"lightgbm":{...},"selected":"mlp"|"lightgbm","n_train":int,"n_test":int,"deployed":"mlp"}`.

### export_onnx.py (pipeline)
- `export_fp32(out=config.MODEL_ONNX_FP32_PATH)` via `torch.onnx.export` with dynamic batch axis (legacy TorchScript exporter).
- `quantize_int8(src=config.MODEL_ONNX_FP32_PATH, dst=config.MODEL_ONNX_INT8_PATH)` via `onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QInt8)`.
- `eval_onnx(path, X, y) -> float` (MAE).
- `main()`: export fp32 + int8, assert int8 size < `config.MODEL_SIZE_BUDGET_MB`, print fp32 vs int8 MAE.

### runtime.py (application lane; MUST NOT import torch)
- `predict_raw(df: pd.DataFrame, onnx_path=config.MODEL_ONNX_INT8_PATH) -> np.ndarray` shape (n,): `preprocess.transform(df[config.ALL_FEATURES])` -> cached `onnxruntime.InferenceSession.run`.
- `get_session(path)` and `get_preprocessor()` cached loaders.
- `predict_profile(profile: dict) -> float`.

### counterfactual.py (application lane)
- `class ModelWrapper: __init__(self, predict_fn); predict(self, X: pd.DataFrame) -> np.ndarray`.
- `build(train_df, predict_fn) -> (dice_ml.Data, dice_ml.Model)` using `Data(dataframe=train_df_with_target, continuous_features=config.NUMERIC_FEATURES, categorical_features=config.CATEGORICAL_FEATURES, outcome_name=config.TARGET)`, `Model(wrapper, backend="sklearn", model_type="regressor")`.
- `suggest(profile: dict, target: float = config.DEFAULT_TARGET_THRESHOLD, total_cfs=config.CF_TOTAL, method=config.CF_METHOD, sparsity=config.CF_SPARSITY) -> list[dict]`.
  Enforce: `features_to_vary=config.ACTIONABLE_FEATURES`, `permitted_range=config.PERMITTED_RANGE`, `desired_range=[target, training target max]`, `posthoc_sparsity_param=sparsity`, `posthoc_sparsity_algorithm="linear"`.
  Return up to `total_cfs` dicts sorted by `(met_target desc, num_changes, distance)`: `{"counterfactual": {feat: val}, "prediction": float, "changes": {feat: [old,new]}, "num_changes": int, "distance": float, "met_target": bool}`.
  Guarantees: immutables unchanged; continuous actionable within permitted range; candidates distinct.

### cli.py (application lane)
- `python -m smcii.cli suggest --profile <json> --target <float> [--json]`; profile json = dict of `config.ALL_FEATURES`. Prints current prediction + suggestions.

### benchmark.py (application lane)
- `latency(n=100, warmup=5) -> {"p50_ms","p95_ms","p99_ms","mean_ms"}` for one `suggest()` call.
- `memory_peak_mb() -> float` (peak RSS via `resource.getrusage`).
- `payload_sizes() -> {"onnx_int8_mb": float, "preprocessor_kb": float}`.
- `run()` writes `config.BENCHMARK_REPORT_PATH` with the above + pass/fail flags vs `LATENCY_BUDGET_MS`, `MEMORY_BUDGET_MB`, `MODEL_SIZE_BUDGET_MB`.

### Tests
Each lane owns `tests/test_<lane>.py`; the integration lane owns `scripts/verify_pipeline.py` and the end-to-end report.
