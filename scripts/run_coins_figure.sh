#!/bin/bash

# Edit before running:
#   1. Set --partition to your SLURM GPU partition.
#   2. Set ENV to your conda environment name.
#   3. Optional: set WANDB_MODE=online if W&B is configured on the cluster.

#SBATCH --job-name=coins_fig
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --account=lingo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/coins_fig_%j.log
#SBATCH --error=slurm_logs/coins_fig_%j.log

ENV="CHANGE_ME"

CKPTS="${CKPTS:-manifest}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

set -euo pipefail

if [ "$ENV" = "CHANGE_ME" ]; then
    echo "Edit ENV in $0 to your conda env name first." >&2
    exit 1
fi

if [ ! -d "$REPO/configs" ]; then
    echo "REPO=$REPO does not look like the Self-CTRL checkout (no configs/)." >&2
    echo "Submit from the repo root, or set REPO=/path/to/Self-CTRL." >&2
    exit 1
fi

cd "$REPO"
mkdir -p slurm_logs

# .bashrc and conda's shell function both reference unset variables, which
# aborts the job under `set -u`.
set +u
source ~/.bashrc
conda activate "$ENV"
set -u

# wandb logging is REQUIRED by default (training and the figure eval refuse to
# run if it can't record). WANDB_MODE=offline is the sanctioned way to run
# without network/credentials — it still records locally for later `wandb sync`.
export WANDB_MODE=${WANDB_MODE:-online}

MANIFEST=configs/figures/coins.yaml

# Each stage's gate checkpoint is derived FROM the figure manifest, so "is this
# stage done?" and "which checkpoint does the figure evaluate?" can never
# disagree. Prints models_root/<run_dir>/ckpt_<n> for the given run_dir slug.
gate_ckpt () {
    python - "$MANIFEST" "$1" <<'PY'
import sys

import yaml

manifest_path, slug = sys.argv[1], sys.argv[2]
with open(manifest_path) as f:
    mf = yaml.safe_load(f)
for cond in mf.get("conditions", []):
    if cond.get("run_dir") == slug:
        models_root = mf.get("models_root", "models/coins")
        print(f"{models_root}/{slug}/ckpt_{int(cond['ckpt'])}")
        sys.exit(0)
sys.exit(f"no condition with run_dir={slug!r} in {manifest_path}")
PY
}

# Resolve every gate up front; under `set -e` a failing assignment aborts here,
# before any training starts.
SFT_GATE="$(gate_ckpt sft)"
CONS_GATE="$(gate_ckpt cons_bw0.0_cont0.3)"
ORACLE_GATE="$(gate_ckpt oracle)"

stage () {
    local name="$1" ckpt="$2"
    shift 2
    local save_dir
    save_dir="$(dirname "$ckpt")"
    if [ -d "$ckpt" ]; then
        echo "== [$name] gate checkpoint $ckpt exists, skipping"
    elif [ -d "$save_dir" ] && [ -n "$(ls -A "$save_dir")" ]; then
        # Training from scratch would interleave new ckpt_<n> dirs with the old
        # ones and silently corrupt the run.
        echo "== [$name] partial run present; refusing to overwrite — inspect or remove $save_dir (gate checkpoint $ckpt is missing)" >&2
        exit 1
    else
        echo "== [$name]"
        "$@"
    fi
}

stage sft "$SFT_GATE" \
    python domains/coins/sft.py

stage cons "$CONS_GATE" \
    python domains/coins/train.py

stage oracle "$ORACLE_GATE" \
    python domains/coins/sft.py --config-name coins_sft_oracle

EVAL_MANIFEST="$MANIFEST"
if [ "$CKPTS" = "last" ]; then
    EVAL_MANIFEST=slurm_logs/coins_resolved.yaml
    python scripts/resolve_ckpts.py "$MANIFEST" "$EVAL_MANIFEST"
fi

python -m evals.coins.plot_calibration --manifest "$EVAL_MANIFEST"