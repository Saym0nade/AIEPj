from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from smcii import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATASET)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep_cols = config.ALL_FEATURES + [config.TARGET]
    df = df[keep_cols].copy()

    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    df = df.dropna(subset=keep_cols)
    return df


def split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = clean(load_raw())
    else:
        df = clean(df)

    train_df = df.sample(frac=1 - test_size, random_state=seed)
    test_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y
