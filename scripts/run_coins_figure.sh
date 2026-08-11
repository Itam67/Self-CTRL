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

# Set WANDB_MODE=online to log runs to W&B directly or offline to avoid requiring credentials.
export WANDB_MODE=${WANDB_MODE:-online}

stage () {
    local name="$1" ckpt="$2"
    shift 2
    if [ -d "$ckpt" ]; then
        echo "== [$name] checkpoint exists, skipping"
    else
        echo "== [$name]"
        "$@"
    fi
}

stage sft models/coins/sft/ckpt_29000 \
    python domains/coins/sft.py

stage cons models/coins/cons_bw0.0_cont0.3/ckpt_1100 \
    python domains/coins/train.py

stage oracle models/coins/oracle/ckpt_43400 \
    python domains/coins/sft.py --config-name coins_sft_oracle

MANIFEST=configs/figures/coins.yaml
if [ "$CKPTS" = "last" ]; then
    MANIFEST=slurm_logs/coins_resolved.yaml
    python scripts/resolve_ckpts.py configs/figures/coins.yaml "$MANIFEST"
fi

python -m evals.coins.plot_calibration --manifest "$MANIFEST"