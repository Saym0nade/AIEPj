"""Tests for the counterfactual engine (smcii.counterfactual)."""

from __future__ import annotations

import numpy as np
import pytest

from smcii import config, counterfactual, data, runtime


@pytest.fixture(scope="module")
def below_threshold_profile():
    _, test_df = data.split()
    below = test_df[test_df[config.TARGET] < config.DEFAULT_TARGET_THRESHOLD]
    assert len(below) > 0, "no below-threshold test profiles"
    return below[config.ALL_FEATURES].iloc[0].to_dict()


@pytest.fixture(scope="module")
def suggestions(below_threshold_profile):
    cfs = counterfactual.suggest(
        below_threshold_profile, target=config.DEFAULT_TARGET_THRESHOLD
    )
    assert len(cfs) > 0, "suggest returned an empty list"
    return cfs


def test_suggest_returns_valid_nonempty_list(suggestions):
    assert isinstance(suggestions, list)
    assert 0 < len(suggestions) <= config.CF_TOTAL
    for cf in suggestions:
        assert set(cf.keys()) == {
            "counterfactual",
            "prediction",
            "changes",
            "num_changes",
            "distance",
            "met_target",
        }


def test_immutables_never_changed(below_threshold_profile, suggestions):
    for cf in suggestions:
        for feat in config.IMMUTABLE_FEATURES:
            assert str(cf["counterfactual"][feat]) == str(below_threshold_profile[feat]), (
                f"immutable {feat} changed"
            )


def test_actionable_within_bounds(below_threshold_profile, suggestions):
    for cf in suggestions:
        for feat, (lo, hi) in config.ACTIONABLE_CONTINUOUS.items():
            val = float(cf["counterfactual"][feat])
            assert lo - 1e-6 <= val <= hi + 1e-6, f"{feat}={val} out of [{lo},{hi}]"


def test_prediction_recomputed_correctly(suggestions):
    for cf in suggestions:
        recomputed = runtime.predict_profile(cf["counterfactual"])
        assert abs(recomputed - cf["prediction"]) < 1e-3


def test_met_target_consistent(suggestions):
    for cf in suggestions:
        assert cf["met_target"] == (cf["prediction"] >= config.DEFAULT_TARGET_THRESHOLD)


def test_changes_match_counterfactual(below_threshold_profile, suggestions):
    for cf in suggestions:
        assert cf["num_changes"] == len(cf["changes"])
        for feat, (old, new) in cf["changes"].items():
            assert feat in config.ACTIONABLE_FEATURES
            assert str(old) == str(below_threshold_profile[feat])
            assert str(new) == str(cf["counterfactual"][feat])


def test_candidates_distinct(suggestions):
    keys = [
        tuple(str(cf["counterfactual"][f]) for f in config.ALL_FEATURES)
        for cf in suggestions
    ]
    assert len(keys) == len(set(keys))


def test_suggest_validates_profile_keys():
    with pytest.raises(ValueError):
        counterfactual.suggest({"Age": 18})