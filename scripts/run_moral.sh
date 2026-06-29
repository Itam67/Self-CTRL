#!/bin/bash
# Launch the moral consistency run (configs/moral.yaml, default explanation run, bw=0.0).
#
# Works two ways:
#   sbatch scripts/run_moral.sh                 # submit as a SLURM job (reads #SBATCH below)
#   ./scripts/run_moral.sh                       # run directly on a GPU node (#SBATCH lines ignored)
# Extra args are forwarded as hydra overrides, e.g.:
#   sbatch scripts/run_moral.sh learning.lr=2e-5 learning.epochs=1
#
#SBATCH --job-name=moral_expl
#SBATCH --partition=<YOUR_GPU_PARTITION>     # e.g. gpu, a100, lingo-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/moral_expl_%j.log
#SBATCH --error=slurm_logs/moral_expl_%j.log

# ---- Edit these ----
REPO=/data/lingo/ipres/projects/Self-CTRL
ENV=<YOUR_CONDA_ENV>
# --------------------

set -euo pipefail

cd "$REPO"
mkdir -p slurm_logs

# Activate the env (conda or venv — keep whichever applies)
source ~/.bashrc
conda activate "$ENV"

# Keep wandb from blocking on login in a batch job; set WANDB_MODE=disabled for smoke tests.
export WANDB_MODE=${WANDB_MODE:-online}

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Started: $(date)"
nvidia-smi || true

python domains/moral.py "$@"

echo "Finished: $(date)"
