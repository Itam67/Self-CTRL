#!/bin/bash
# Launch a moral consistency run. Submit from the repo root.
#
# Works two ways:
#   sbatch scripts/run_moral.sh                 # submit as a SLURM job (reads #SBATCH below)
#   ./scripts/run_moral.sh                      # run directly on a GPU node (#SBATCH lines ignored)
#
# Which run: pass a config name (default configs/moral_expl.yaml — λ=0).
#   sbatch scripts/run_moral.sh --config-name moral_mixed          # λ=0.5
#   sbatch scripts/run_moral.sh --config-name moral_beh            # λ=1.0
# Set ENTRY for the introspective baselines:
#   ENTRY=domains/moral/baseline.py sbatch scripts/run_moral.sh    # Expl. baseline
#   ENTRY=domains/moral/baseline.py sbatch scripts/run_moral.sh --config-name moral_baseline_beh
#
# Any other argument is forwarded as a hydra override, e.g.:
#   sbatch scripts/run_moral.sh learning.lr=2e-5 learning.epochs=1
#
#SBATCH --job-name=moral
#SBATCH --partition=<YOUR_GPU_PARTITION>     # e.g. gpu, a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/moral_%j.log
#SBATCH --error=slurm_logs/moral_%j.log

# ---- Edit this ----
ENV="CHANGE_ME"   # your conda env name
# -------------------

# Repo root: the directory sbatch was submitted from (SLURM starts the job
# there), or the current directory for a direct run. Override with REPO=... if
# you submit from somewhere else.
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

# Activate the env (conda or venv — keep whichever applies)
source ~/.bashrc
conda activate "$ENV"

# Keep wandb from blocking on login in a batch job; set WANDB_MODE=disabled for smoke tests.
export WANDB_MODE=${WANDB_MODE:-online}

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Started: $(date)"
nvidia-smi || true

# ENTRY selects the trainer; the introspective baselines share this launcher:
#   ENTRY=domains/moral/baseline.py sbatch scripts/run_moral.sh learning.bw=1.0
python "${ENTRY:-domains/moral/train.py}" "$@"

echo "Finished: $(date)"
