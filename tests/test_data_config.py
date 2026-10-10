"""Independent QA checks for the frozen data/config/preprocess contract.

Run with:
    PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_data_config.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from smcii import config, data, preprocess


def test_raw_csv_has_expected_16_columns():
    df = data.load_raw()
    expected = set(config.ID_COLUMNS + config.ALL_FEATURES + [config.TARGET] + ["Overall_Impact"])
    assert len(df.columns) == 16
    assert set(df.columns) == expected
    # every model feature + the target must be present
    assert set(config.ALL_FEATURES + [config.TARGET]).issubset(set(df.columns))


def test_all_features_is_disjoint_union_of_immutable_and_actionable():
    all_f = list(config.ALL_FEATURES)
    immut = set(config.IMMUTABLE_FEATURES)
    act = set(config.ACTIONABLE_FEATURES)

    assert len(all_f) == len(set(all_f)), "ALL_FEATURES contains duplicates"
    assert immut.isdisjoint(act), "immutable and actionable overlap"
    assert immut | act == set(all_f), "ALL_FEATURES != IMMUTABLE | ACTIONABLE"
    # no feature is silently missing/extra in either role list
    assert set(all_f) == immut.union(act)


def test_actionable_continuous_bounds_sane_and_match_permitted_range():
    assert config.ACTIONABLE_CONTINUOUS, "no actionable continuous features declared"
    for feat, (lo, hi) in config.ACTIONABLE_CONTINUOUS.items():
        assert lo < hi, f"{feat}: bad bounds ({lo}, {hi})"
    assert config.PERMITTED_RANGE == {
        k: [float(lo), float(hi)] for k, (lo, hi) in config.ACTIONABLE_CONTINUOUS.items()
    }
    assert set(config.PERMITTED_RANGE) == set(config.ACTIONABLE_CONTINUOUS)


def test_clean_drops_null_rows_and_keeps_only_features_and_target():
    raw = data.load_raw()
    cleaned = data.clean(raw)

    keep = config.ALL_FEATURES + [config.TARGET]
    assert list(cleaned.columns) == keep
    assert cleaned[keep].isnull().sum().sum() == 0

    dropped = len(raw) - len(cleaned)
    assert dropped > 0, "expected some rows with nulls to be dropped"
    assert dropped <= len(raw)
    print(f"[data.clean] dropped {dropped} of {len(raw)} raw rows ({len(cleaned)} kept)")

    # a row with an injected null in a model column must be dropped
    valid = cleaned.head(20).copy()
    dirty = valid.copy()
    dirty.loc[dirty.index[0], config.TARGET] = np.nan
    assert len(data.clean(dirty)) == len(valid) - 1


def test_split_is_deterministic_and_80_20():
    cleaned = data.clean(data.load_raw())
    n = len(cleaned)

    tr1, te1 = data.split(seed=config.RANDOM_SEED)
    tr2, te2 = data.split(seed=config.RANDOM_SEED)

    # same seed -> identical split contents
    assert tr1.equals(tr2)
    assert te1.equals(te2)

    # different seed -> different ordering (determinism is meaningful)
    tr_alt, _ = data.split(seed=config.RANDOM_SEED + 1)
    assert not tr_alt.equals(tr1), "different seeds produced identical split"

    # sizes partition the data 80/20
    assert len(tr1) + len(te1) == n
    assert abs(len(te1) / n - config.TEST_SIZE) < 0.01
    assert abs(len(tr1) / n - (1 - config.TEST_SIZE)) < 0.01

    # train/test are a complete partition of the cleaned data (no loss/dup)
    cols = config.ALL_FEATURES + [config.TARGET]
    combined = (
        pd.concat([tr1, te1])[cols]
        .sort_values(by=cols)
        .reset_index(drop=True)
    )
    expected_partition = cleaned[cols].sort_values(by=cols).reset_index(drop=True)
    assert combined.equals(expected_partition)


def test_transform_single_reordered_row_matches_metadata():
    with open(config.FEATURE_METADATA_PATH) as f:
        meta = json.load(f)

    _, test_df = data.split()
    row = test_df[config.ALL_FEATURES].iloc[[0]]
    reordered = row[config.ALL_FEATURES[::-1]]

    out = preprocess.transform(reordered)
    assert out.ndim == 2
    assert out.shape == (1, meta["n_features_out"])
    assert out.dtype == np.float32


def test_transform_accepts_series_and_keeps_column_alignment():
    _, test_df = data.split()
    row_series = test_df[config.ALL_FEATURES].iloc[0]

    out = preprocess.transform(row_series)
    assert out.shape == (1, 31)
    assert out.dtype == np.float32
