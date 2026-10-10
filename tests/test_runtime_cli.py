"""Tests for the ONNX runtime and CLI (no torch imports)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from smcii import config, data, runtime


def test_runtime_module_does_not_import_torch():
    code = (
        "import sys; "
        "from smcii import runtime; "
        "assert 'torch' not in sys.modules, 'torch imported by runtime'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": "src"},
    )


def test_predict_raw_finite_in_range_and_batch_shaped():
    _, test_df = data.split()
    X, _ = data.X_y(test_df)
    preds = runtime.predict_raw(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert preds.dtype == np.float32
    assert np.isfinite(preds).all()
    assert preds.min() >= 0.0
    assert preds.max() <= 120.0


def test_predict_profile_matches_predict_raw():
    _, test_df = data.split()
    profile = test_df[config.ALL_FEATURES].iloc[0].to_dict()
    single = runtime.predict_profile(profile)
    batch = runtime.predict_raw(test_df[config.ALL_FEATURES].iloc[[0]])
    assert abs(single - float(batch[0])) < 1e-4


def test_cli_suggest_output(tmp_path):
    from smcii import cli

    _, test_df = data.split()
    profile = test_df[config.ALL_FEATURES].iloc[0].to_dict()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    rc = cli.main(["suggest", "--profile", str(profile_path), "--target", "75"])
    assert rc == 0


def test_cli_suggest_json_output(tmp_path, capsys):
    from smcii import cli

    _, test_df = data.split()
    profile = test_df[config.ALL_FEATURES].iloc[0].to_dict()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    rc = cli.main(["suggest", "--profile", str(profile_path), "--target", "75", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "current_prediction" in out
    assert "counterfactuals" in out


def test_cli_rejects_bad_profile(tmp_path):
    from smcii import cli

    profile_path = tmp_path / "bad.json"
    profile_path.write_text(json.dumps({"Age": 18}))

    with pytest.raises(SystemExit) as exc:
        cli.main(["suggest", "--profile", str(profile_path), "--target", "75"])
    assert exc.value.code != 0