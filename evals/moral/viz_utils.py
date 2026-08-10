"""JSON eval-snapshot loaders for the consistency-curves figure."""

import json
from pathlib import Path


def load_all_snapshots(save_dir: Path):
    """Load all eval snapshot JSONs, keyed by step number.

    Normalizes legacy `avg_judge_score` / per-record `judge_score` keys to the
    current `avg_consistency` / `consistency_score` names.
    """
    snap_dir = save_dir / "eval_snapshots"
    files = sorted(
        snap_dir.glob("eval_*.json"), key=lambda p: int(p.stem.split("_")[1])
    )
    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 eval snapshots in {snap_dir}, found {len(files)}. "
            "Make sure training ran long enough for at least 2 evals."
        )
    snapshots = {}
    for f_path in files:
        with open(f_path) as f:
            data = json.load(f)
        step = data["examples_seen"]
        if "avg_consistency" not in data and "avg_judge_score" in data:
            data["avg_consistency"] = data["avg_judge_score"]
        for r in data.get("records", []):
            if "consistency_score" not in r and "judge_score" in r:
                r["consistency_score"] = r["judge_score"]
        snapshots[step] = data
    return snapshots


def load_holdout_snapshots(save_dir: Path):
    """Load all holdout_eval_*.json snapshots, keyed by step number. Returns {} if none found.

    Normalizes legacy `holdout_avg_judge_score` / per-record `judge_score` keys to
    the current `avg_consistency` / `consistency_score` names.
    """
    snap_dir = save_dir / "eval_snapshots"
    files = sorted(
        snap_dir.glob("holdout_eval_*.json"), key=lambda p: int(p.stem.split("_", 2)[2])
    )
    snapshots = {}
    for f_path in files:
        with open(f_path) as f:
            data = json.load(f)
        step = data["examples_seen"]
        # Top-level normalization
        if "avg_consistency" not in data:
            if "holdout_avg_consistency" in data:
                data["avg_consistency"] = data["holdout_avg_consistency"]
            elif "holdout_avg_judge_score" in data:
                data["avg_consistency"] = data["holdout_avg_judge_score"]
            elif "avg_judge_score" in data:
                data["avg_consistency"] = data["avg_judge_score"]
        # Per-record normalization (legacy snapshots have judge_score)
        for r in data.get("records", []):
            if "consistency_score" not in r and "judge_score" in r:
                r["consistency_score"] = r["judge_score"]
        snapshots[step] = data
    return snapshots


def _split_holdout_snapshots(holdout_snapshots):
    """Split holdout snapshots into broad-category and fine-principle subsets.

    Returns (broad, fine) — each a dict[step -> snapshot-like] that the
    consistency-curve series builder can consume. Steps lacking the holdout_type
    field on records (older runs) are skipped from the split.
    """
    broad, fine = {}, {}
    for step, snap in holdout_snapshots.items():
        records = snap.get("records", [])
        if not records or "holdout_type" not in records[0]:
            continue
        for htype, out in (("broad_category", broad), ("fine_principle", fine)):
            sub = [r for r in records if r.get("holdout_type") == htype]
            if not sub:
                continue
            scores = [r["consistency_score"] for r in sub]
            out[step] = {
                "examples_seen": step,
                "avg_consistency": sum(scores) / len(scores) if scores else 0.0,
                "records": sub,
            }
    return broad, fine


def _series_from_snapshots(snapshots, key="avg_consistency"):
    """Return (steps, values) sorted by step from a snapshot dict, dropping None ys."""
    pairs = sorted(snapshots.items())
    out = [(s, snap.get(key)) for s, snap in pairs]
    out = [(s, y) for s, y in out if y is not None]
    if not out:
        return [], []
    xs, ys = zip(*out)
    return list(xs), list(ys)
