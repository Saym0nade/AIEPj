"""Counterfactual engine (DiCE, sklearn backend wrapping the ONNX predictor).

Builds a DiCE explainer once over the training data and exposes ``suggest``,
which returns the minimal actionable changes that push a profile's predicted
Mental Health Index above a target threshold.

If DiCE is unavailable or fails, an internal KD-Tree / random-search fallback
with the same ``suggest`` contract is used (last resort only).
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from smcii import config, data, runtime

_dice = None
_dice_method = None
_train_df = None
_train_target_max = None
_fallback = None


class ModelWrapper:
    """Adapter exposing a 1-D predict() to DiCE's sklearn backend."""

    def __init__(self, predict_fn: Callable[[pd.DataFrame], np.ndarray]):
        self.predict_fn = predict_fn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predict_fn(X), dtype=np.float64).reshape(-1)


def _get_train():
    global _train_df, _train_target_max
    if _train_df is None:
        _train_df, _ = data.split()
        _train_target_max = float(_train_df[config.TARGET].max())
    return _train_df, _train_target_max


def _build_dice(method: str):
    """Build and cache the DiCE explainer for the given method."""
    global _dice, _dice_method, _train_df, _train_target_max
    if _dice is not None and _dice_method == method:
        return _dice
    import dice_ml

    train_df, _ = _get_train()
    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=list(config.NUMERIC_FEATURES),
        categorical_features=list(config.CATEGORICAL_FEATURES),
        outcome_name=config.TARGET,
    )
    m = dice_ml.Model(
        model=ModelWrapper(runtime.predict_raw),
        backend="sklearn",
        model_type="regressor",
    )
    _dice = dice_ml.Dice(d, m, method=method)
    _dice_method = method
    return _dice


def _validate_profile(profile: dict) -> None:
    missing = [c for c in config.ALL_FEATURES if c not in profile]
    extra = [c for c in profile if c not in config.ALL_FEATURES]
    if missing or extra:
        raise ValueError(
            f"profile must contain exactly config.ALL_FEATURES; "
            f"missing={missing}, extra={extra}"
        )


def _normalized_distance(profile: dict, cf: dict) -> float:
    """Normalized L1 over actionable features (continuous |d|/(hi-lo), cat +1)."""
    dist = 0.0
    for feat in config.ACTIONABLE_FEATURES:
        old = profile[feat]
        new = cf[feat]
        if feat in config.PERMITTED_RANGE:
            lo, hi = config.PERMITTED_RANGE[feat]
            dist += abs(float(new) - float(old)) / (hi - lo)
        else:
            if str(old) != str(new):
                dist += 1.0
    return float(dist)


def _changes(profile: dict, cf: dict) -> dict:
    out = {}
    for feat in config.ACTIONABLE_FEATURES:
        old = profile[feat]
        new = cf[feat]
        if feat in config.PERMITTED_RANGE:
            if abs(float(new) - float(old)) > 1e-6:
                out[feat] = [old, new]
        else:
            if str(old) != str(new):
                out[feat] = [old, new]
    return out


def _is_valid_cf(profile: dict, cf: dict) -> bool:
    """Hard-verify immutables unchanged and actionable within bounds."""
    for feat in config.IMMUTABLE_FEATURES:
        if str(profile[feat]) != str(cf[feat]):
            return False
    for feat in config.ACTIONABLE_FEATURES:
        if feat in config.PERMITTED_RANGE:
            lo, hi = config.PERMITTED_RANGE[feat]
            val = float(cf[feat])
            if not (lo - 1e-6 <= val <= hi + 1e-6):
                return False
        else:
            if str(cf[feat]) not in _categorical_values(feat):
                return False
    return True


def _categorical_values(feat: str) -> set:
    train_df, _ = _get_train()
    return set(train_df[feat].astype(str).unique())


def _postprocess(profile: dict, cf_rows: pd.DataFrame, target: float, total_cfs: int) -> List[dict]:
    results = []
    seen = set()
    for _, row in cf_rows.iterrows():
        cf = {c: row[c] for c in config.ALL_FEATURES}
        if not _is_valid_cf(profile, cf):
            continue
        key = tuple(str(cf[c]) for c in config.ALL_FEATURES)
        if key in seen:
            continue
        seen.add(key)
        pred = float(runtime.predict_profile(cf))
        chg = _changes(profile, cf)
        results.append(
            {
                "counterfactual": cf,
                "prediction": pred,
                "changes": chg,
                "num_changes": len(chg),
                "distance": _normalized_distance(profile, cf),
                "met_target": bool(pred >= target),
            }
        )
    results.sort(
        key=lambda r: (not r["met_target"], r["num_changes"], r["distance"])
    )
    return results[:total_cfs]


def _suggest_dice(profile: dict, target: float, total_cfs: int, method: str, sparsity: float) -> List[dict]:
    train_df, train_max = _get_train()
    dice = _build_dice(method)
    query = pd.DataFrame([profile]).reindex(columns=config.ALL_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand = dice.generate_counterfactuals(
            query,
            total_CFs=total_cfs,
            desired_range=[float(target), train_max],
            features_to_vary=list(config.ACTIONABLE_FEATURES),
            permitted_range=config.PERMITTED_RANGE,
            posthoc_sparsity_param=sparsity,
            posthoc_sparsity_algorithm="linear",
        )
    cf_rows = cand.cf_examples_list[0].final_cfs_df
    if config.TARGET in cf_rows.columns:
        cf_rows = cf_rows.drop(columns=[config.TARGET])
    return _postprocess(profile, cf_rows, target, total_cfs)


class _FallbackEngine:
    """Internal KD-Tree / random-search fallback (last resort only).

    Candidates come from (a) a KD-tree over the training set's normalized
    actionable continuous features and (b) uniform random sampling of the
    actionable space. All candidates keep immutables fixed.
    """

    def __init__(self):
        from scipy.spatial import KDTree

        train_df, _ = _get_train()
        self._train_df = train_df
        cont = config.ACTIONABLE_CONTINUOUS
        self._cont_cols = list(cont.keys())
        self._cat_cols = list(config.ACTIONABLE_CATEGORICAL)
        self._cat_values = {c: sorted(set(train_df[c].astype(str))) for c in self._cat_cols}
        self._rng = np.random.default_rng(config.RANDOM_SEED)

        pts = train_df[self._cont_cols].to_numpy(dtype=np.float64)
        norm = np.empty_like(pts)
        for i, c in enumerate(self._cont_cols):
            lo, hi = cont[c]
            norm[:, i] = (pts[:, i] - lo) / (hi - lo)
        self._tree = KDTree(norm)
        self._norm_pts = norm

    def _sample_candidates(self, profile: dict, n: int) -> List[dict]:
        cont = config.ACTIONABLE_CONTINUOUS
        cands = []
        for _ in range(n):
            cf = dict(profile)
            for c in self._cont_cols:
                lo, hi = cont[c]
                cf[c] = float(self._rng.uniform(lo, hi))
            for c in self._cat_cols:
                cf[c] = str(self._rng.choice(self._cat_values[c]))
            cands.append(cf)
        return cands

    def _neighbor_candidates(self, profile: dict, k: int) -> List[dict]:
        cont = config.ACTIONABLE_CONTINUOUS
        q = np.array(
            [(float(profile[c]) - cont[c][0]) / (cont[c][1] - cont[c][0]) for c in self._cont_cols]
        )
        dists, idxs = self._tree.query(q, k=min(k, len(self._norm_pts)))
        if np.ndim(idxs) == 0:
            idxs = [int(idxs)]
        cands = []
        for i in np.atleast_1d(idxs):
            cf = dict(profile)
            for j, c in enumerate(self._cont_cols):
                cf[c] = float(self._train_df.iloc[int(i)][c])
            for c in self._cat_cols:
                cf[c] = str(self._train_df.iloc[int(i)][c])
            cands.append(cf)
        return cands

    def suggest(self, profile: dict, target: float, total_cfs: int, sparsity: float) -> List[dict]:
        cands = self._neighbor_candidates(profile, k=50) + self._sample_candidates(profile, n=400)
        rows = []
        for cf in cands:
            if not _is_valid_cf(profile, cf):
                continue
            cf = {c: cf[c] for c in config.ALL_FEATURES}
            pred = float(runtime.predict_profile(cf))
            chg = _changes(profile, cf)
            rows.append(
                {
                    "counterfactual": cf,
                    "prediction": pred,
                    "changes": chg,
                    "num_changes": len(chg),
                    "distance": _normalized_distance(profile, cf),
                    "met_target": bool(pred >= target),
                }
            )
        rows.sort(key=lambda r: (not r["met_target"], r["num_changes"], r["distance"]))
        seen, out = set(), []
        for r in rows:
            key = tuple(str(r["counterfactual"][c]) for c in config.ALL_FEATURES)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= total_cfs:
                break
        return out


def _get_fallback() -> _FallbackEngine:
    global _fallback
    if _fallback is None:
        _fallback = _FallbackEngine()
    return _fallback


def suggest(
    profile: dict,
    target: float = config.DEFAULT_TARGET_THRESHOLD,
    total_cfs: int = config.CF_TOTAL,
    method: Optional[str] = None,
    sparsity: float = config.CF_SPARSITY,
) -> List[dict]:
    """Return up to ``total_cfs`` counterfactual suggestions for ``profile``.

    Each result: {"counterfactual": {...}, "prediction": float, "changes": {...},
    "num_changes": int, "distance": float, "met_target": bool}.
    """
    _validate_profile(profile)
    method = method or config.CF_METHOD

    for m in (method, "kdtree"):
        try:
            return _suggest_dice(profile, target, total_cfs, m, sparsity)
        except Exception:
            continue

    return _get_fallback().suggest(profile, target, total_cfs, sparsity)