from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SRC_DIR = PROJECT_ROOT / "src"

RAW_DATASET = DATA_DIR / "social_media_impact.csv"
DATASET_PROFILE = DATA_DIR / "dataset_profile.json"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"
MODEL_TORCH_PATH = ARTIFACTS_DIR / "model.pt"
MODEL_ONNX_FP32_PATH = ARTIFACTS_DIR / "model_fp32.onnx"
MODEL_ONNX_INT8_PATH = ARTIFACTS_DIR / "model_int8.onnx"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
BENCHMARK_REPORT_PATH = ARTIFACTS_DIR / "benchmark_report.json"
INTEGRATION_REPORT_PATH = ARTIFACTS_DIR / "integration_report.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Mental_Health_Index"

ID_COLUMNS = ["Student_ID"]
DROP_COLUMNS = ["Student_ID", "Overall_Impact"]

IMMUTABLE_FEATURES = [
    "Age",
    "Gender",
    "Academic_Level",
    "Sleep_Quality_Score",
    "Social_Comparison_Frequency",
    "Perceived_Stress_Score",
    "Late_Night_Usage",
    "Weekend_Extra_Hours",
    "Academic_Performance_GPA",
]

ACTIONABLE_CONTINUOUS = {
    "Sleep_Duration_Hours": (4.0, 10.0),
    "Daily_Usage_Hours": (0.0, 8.0),
}
ACTIONABLE_CATEGORICAL = ["Primary_Platform", "Device_Type"]
ACTIONABLE_FEATURES = list(ACTIONABLE_CONTINUOUS.keys()) + list(ACTIONABLE_CATEGORICAL)

NUMERIC_FEATURES = [
    "Age",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Academic_Performance_GPA",
    "Sleep_Duration_Hours",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Social_Comparison_Frequency",
    "Late_Night_Usage",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PERMITTED_RANGE = {k: [float(lo), float(hi)] for k, (lo, hi) in ACTIONABLE_CONTINUOUS.items()}

DEFAULT_TARGET_THRESHOLD = 75.0
CF_TOTAL = 3
CF_METHOD = "random"
CF_SPARSITY = 0.1

LATENCY_BUDGET_MS = 200.0
MEMORY_BUDGET_MB = 1024.0
MODEL_SIZE_BUDGET_MB = 5.0

MLP_HIDDEN = (64, 32)
MLP_EPOCHS = 300
MLP_LR = 0.001
MLP_BATCH = 64
MLP_PATIENCE = 30
