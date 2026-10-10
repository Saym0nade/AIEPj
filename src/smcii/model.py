from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from smcii import config, data, preprocess


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def _set_seeds():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)


def _train_one(
    Xtr_np: np.ndarray,
    ytr_np: np.ndarray,
    Xval_np: Optional[np.ndarray] = None,
    yval_np: Optional[np.ndarray] = None,
) -> MLP:
    """Train one MLP. The target is standardized for training, then the
    de-normalization is folded into the final linear layer so the returned
    model predicts in ORIGINAL target units."""
    _set_seeds()
    model = MLP(Xtr_np.shape[1])
    y_mean = float(np.mean(ytr_np))
    y_std = float(np.std(ytr_np)) or 1.0

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR, weight_decay=1e-5)

    Xtr = torch.from_numpy(Xtr_np)
    ytr = torch.from_numpy(((ytr_np - y_mean) / y_std).astype(np.float32)).view(-1, 1)
    has_val = Xval_np is not None and yval_np is not None
    if has_val:
        Xval = torch.from_numpy(Xval_np)
        yval = torch.from_numpy(((yval_np - y_mean) / y_std).astype(np.float32)).view(-1, 1)

    best = float("inf")
    patience = 0
    best_state = None
    for _ in range(config.MLP_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            monitor = criterion(
                model(Xval) if has_val else model(Xtr),
                yval if has_val else ytr,
            ).item()
        if monitor < best:
            best = monitor
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.MLP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # bake de-normalization into the last linear layer -> original units
    with torch.no_grad():
        model.net[-1].weight.mul_(y_std)
        model.net[-1].bias.mul_(y_std).add_(y_mean)
    model.eval()
    return model


def _global_test() -> pd.DataFrame:
    _, test_df = data.split()
    return test_df


def train_mlp(train_df: pd.DataFrame) -> Tuple[MLP, dict]:
    torch.set_num_threads(4)

    train_df = train_df.reset_index(drop=True)
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    cv_maes = []

    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]

        preproc = preprocess.build()
        Xf_tr, yf_tr = data.X_y(fold_train)
        Xf_val, yf_val = data.X_y(fold_val)
        preproc.fit(Xf_tr)
        Xtr_t = preproc.transform(Xf_tr).astype(np.float32)
        Xval_t = preproc.transform(Xf_val).astype(np.float32)

        fold_model = _train_one(
            Xtr_t,
            yf_tr.values.astype(np.float32),
            Xval_t,
            yf_val.values.astype(np.float32),
        )
        with torch.no_grad():
            pred = fold_model(torch.from_numpy(Xval_t)).numpy().flatten()
        cv_maes.append(mean_absolute_error(yf_val.values, pred))

    cv_mae = float(np.mean(cv_maes))
    cv_mae_std = float(np.std(cv_maes))

    # persisted preprocessor + final model on all of train
    preproc_final = preprocess.fit_save(train_df)
    X_tr, y_tr = data.X_y(train_df)
    X_tr_t = preproc_final.transform(X_tr).astype(np.float32)

    # internal monitor split for early stopping
    fit_df = train_df.sample(frac=0.9, random_state=config.RANDOM_SEED)
    mon_df = train_df.drop(fit_df.index)
    X_fit, y_fit = data.X_y(fit_df)
    X_mon, y_mon = data.X_y(mon_df)
    X_fit_t = preproc_final.transform(X_fit).astype(np.float32)
    X_mon_t = preproc_final.transform(X_mon).astype(np.float32)

    model_final = _train_one(
        X_fit_t,
        y_fit.values.astype(np.float32),
        X_mon_t,
        y_mon.values.astype(np.float32),
    )
    torch.save(
        {"state_dict": model_final.state_dict(), "n_features": X_tr_t.shape[1]},
        config.MODEL_TORCH_PATH,
    )

    test_df = _global_test()
    X_test, y_test = data.X_y(test_df)
    X_test_t = preproc_final.transform(X_test).astype(np.float32)
    with torch.no_grad():
        y_pred_test = model_final(torch.from_numpy(X_test_t)).numpy().flatten()
    mlp_holdout_mae = float(mean_absolute_error(y_test.values, y_pred_test))
    mlp_rmse = float(np.sqrt(mean_squared_error(y_test.values, y_pred_test)))

    metrics = _load_metrics()
    metrics["mlp"] = {
        "cv_mae": cv_mae,
        "cv_mae_std": cv_mae_std,
        "holdout_mae": mlp_holdout_mae,
        "rmse": mlp_rmse,
    }
    metrics.setdefault("lightgbm", {})
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    metrics["deployed"] = "mlp"
    _select(metrics)
    _write_metrics(metrics)

    return model_final, metrics


def train_lgbm(train_df: pd.DataFrame) -> Tuple[LGBMRegressor, dict]:
    preproc = preprocess.load()
    X_train, y_train = data.X_y(train_df)
    X_train_t = preproc.transform(X_train).astype(np.float32)

    model = LGBMRegressor(
        random_state=config.RANDOM_SEED,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
    )
    model.fit(X_train_t, y_train.values)

    test_df = _global_test()
    X_test, y_test = data.X_y(test_df)
    X_test_t = preproc.transform(X_test).astype(np.float32)
    y_pred = model.predict(X_test_t)
    holdout_mae = float(mean_absolute_error(y_test.values, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test.values, y_pred)))

    metrics = _load_metrics()
    metrics["lightgbm"] = {"holdout_mae": holdout_mae, "rmse": rmse}
    metrics["deployed"] = "mlp"
    _select(metrics)
    _write_metrics(metrics)
    return model, metrics


def _load_metrics() -> dict:
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return {}


def _write_metrics(metrics: dict) -> None:
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def _select(metrics: dict) -> None:
    mlp_mae = metrics.get("mlp", {}).get("holdout_mae")
    lgbm_mae = metrics.get("lightgbm", {}).get("holdout_mae")
    if mlp_mae is None:
        metrics["selected"] = "lightgbm"
    elif lgbm_mae is None:
        metrics["selected"] = "mlp"
    else:
        metrics["selected"] = "mlp" if mlp_mae <= lgbm_mae else "lightgbm"
