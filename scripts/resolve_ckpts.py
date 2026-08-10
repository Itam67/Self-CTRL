"""Rewrite a figure manifest's `ckpt:` values to the highest checkpoint on disk.

Used by the run_*_figure.sh launchers under CKPTS=last, so a deck can be plotted
from whatever training actually produced without editing the tracked manifest.

    python scripts/resolve_ckpts.py <in.yaml> <out.yaml>
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.manifest import REPO_ROOT, get_checkpoint_dirs  # noqa: E402


def main(src, dst):
    raw = yaml.safe_load(Path(src).read_text())
    models_root = Path(raw.get("models_root", "models/moral"))
    if not models_root.is_absolute():
        models_root = REPO_ROOT / models_root

    for entry in raw.get("conditions", []):
        run_dir = entry.get("run_dir")
        if run_dir is None:
            continue  # the untrained base model has no checkpoint
        path = models_root / run_dir
        if not path.is_dir():
            print(f"  {entry['label']!r}: {path} missing, leaving ckpt as-is")
            continue
        ckpts = get_checkpoint_dirs(path)
        if not ckpts:
            print(f"  {entry['label']!r}: no ckpt_<n> in {path}, leaving ckpt as-is")
            continue
        last = ckpts[-1][0]
        was = entry.get("ckpt")
        entry["ckpt"] = last
        print(f"  {entry['label']!r}: ckpt {was} -> {last}")

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
