"""Command-line interface for counterfactual suggestions.

Usage:
    python -m smcii.cli suggest --profile <path.json> --target <float> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from smcii import config, counterfactual, runtime


def _load_profile(path: str) -> dict:
    try:
        with open(path) as f:
            profile = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"error: profile file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    if not isinstance(profile, dict):
        raise SystemExit("error: profile must be a JSON object")

    missing = [c for c in config.ALL_FEATURES if c not in profile]
    extra = [c for c in profile if c not in config.ALL_FEATURES]
    if missing or extra:
        raise SystemExit(
            f"error: profile must contain exactly config.ALL_FEATURES; "
            f"missing={missing}, extra={extra}"
        )
    return profile


def _suggest(profile_path: str, target: float, as_json: bool) -> int:
    profile = _load_profile(profile_path)
    current = runtime.predict_profile(profile)

    if as_json:
        out = {
            "profile": profile,
            "current_prediction": current,
            "target": target,
            "counterfactuals": counterfactual.suggest(profile, target=target),
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Current prediction: {current:.2f}")
    print(f"Target threshold:   {target:.2f}")
    print()
    cfs = counterfactual.suggest(profile, target=target)
    if not cfs:
        print("No counterfactuals found.")
        return 0
    for i, cf in enumerate(cfs, start=1):
        print(f"--- Counterfactual {i} ---")
        print(f"  predicted: {cf['prediction']:.2f}  met_target: {cf['met_target']}")
        if cf["changes"]:
            for feat, (old, new) in cf["changes"].items():
                print(f"  {feat}: {old} -> {new}")
        else:
            print("  (no actionable changes)")
        print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smcii.cli", description="smcii counterfactual CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("suggest", help="suggest counterfactual changes")
    p.add_argument("--profile", required=True, help="path to profile JSON")
    p.add_argument("--target", type=float, default=config.DEFAULT_TARGET_THRESHOLD)
    p.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)
    if args.command == "suggest":
        return _suggest(args.profile, args.target, args.json)
    return 1


if __name__ == "__main__":
    sys.exit(main())