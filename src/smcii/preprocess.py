from __future__ import annotations

import json
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from smcii import config, data


def build() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_save(train_df: pd.DataFrame) -> ColumnTransformer:
    X, _ = data.X_y(train_df)
    preprocessor = build()
    preprocessor.fit(X)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    feature_names_out = preprocessor.get_feature_names_out()
    ohe = preprocessor.named_transformers_["cat"]
    ohe_categories = {}
    for i, col in enumerate(config.CATEGORICAL_FEATURES):
        cats = list(ohe.categories_[i])
        ohe_categories[col] = cats

    metadata = {
        "numeric": list(config.NUMERIC_FEATURES),
        "categorical": list(config.CATEGORICAL_FEATURES),
        "ohe_categories": ohe_categories,
        "n_features_in": len(config.ALL_FEATURES),
        "n_features_out": len(feature_names_out),
        "feature_names_out": list(feature_names_out),
        "actionable": list(config.ACTIONABLE_FEATURES),
        "immutable": list(config.IMMUTABLE_FEATURES),
        "permitted_range": {k: [float(lo), float(hi)] for k, (lo, hi) in config.ACTIONABLE_CONTINUOUS.items()},
    }
    config.FEATURE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.FEATURE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return preprocessor


def load() -> ColumnTransformer:
    return joblib.load(config.PREPROCESSOR_PATH)


def transform(df: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    df = df.copy()
    if not all(col in df.columns for col in config.ALL_FEATURES):
        raise ValueError("DataFrame missing required feature columns")

    df = df.reindex(columns=config.ALL_FEATURES, fill_value=np.nan)
    preprocessor = load()
    X_trans = preprocessor.transform(df)
    return X_trans.astype(np.float32)
