#!/bin/bash

# Edit before running:
#   1. Set --partition to your SLURM GPU partition.
#   2. Set ENV to your conda environment name.
#   3. Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT to your GCP project.
#   4. Optional: set WANDB_MODE=online if W&B is configured on the cluster.
#SBATCH --job-name=moral_figs
#SBATCH --partition=<CHANGE_ME>
#SBATCH --qos=<CHANGE_ME>
#SBATCH --account=<CHANGE_ME>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/moral_figs_%j.log
#SBATCH --error=slurm_logs/moral_figs_%j.log

ENV="CHANGE_ME"
GCP_PROJECT="${GOOGLE_CLOUD_PROJECT:-CHANGE_ME}"

TRAIN="${TRAIN:-1}"
CKPTS="${CKPTS:-manifest}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

set -euo pipefail

if [ "$ENV" = "CHANGE_ME" ]; then
    echo "Edit ENV in $0 to your conda env name first." >&2
    exit 1
fi

if [ "$GCP_PROJECT" = "CHANGE_ME" ]; then
    echo "Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT in $0." >&2
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

export GOOGLE_CLOUD_PROJECT="$GCP_PROJECT"
# Set WANDB_MODE=online to log runs to W&B directly or offline to avoid requiring credentials.
export WANDB_MODE=${WANDB_MODE:-online}

python - <<'PY'
from evals.gemini import get_gemini_client
get_gemini_client()
print("Vertex ADC OK")
PY

run () {
    local name="$1" ckpt="$2" entry="$3"
    shift 3

    if [ "$TRAIN" != "1" ]; then
        echo "== [$name] TRAIN=0, skipping"
    elif [ -d "$ckpt" ]; then
        echo "== [$name] checkpoint exists, skipping"
    else
        echo "== [$name]"
        python "$entry" "$@"
    fi
}

M=models/moral

run expl \
    "$M/bw0.0_cont0.0_aux0.0_ukl0.1/ckpt_2808" \
    domains/moral/train.py

run mixed \
    "$M/bw0.5_cont0.25_aux0.4_ukl0.0/ckpt_3604" \
    domains/moral/train.py --config-name moral_mixed

run beh \
    "$M/bw1.0_cont0.25_aux0.4_ukl0.1/ckpt_3604" \
    domains/moral/train.py --config-name moral_beh

run expl_baseline \
    "$M/baseline_expl_cont0.25_ukl0.1/ckpt_2202" \
    domains/moral/baseline.py

run beh_baseline \
    "$M/baseline_beh_cont0.25_ukl0.1/ckpt_3404" \
    domains/moral/baseline.py --config-name moral_baseline_beh

MANIFEST=configs/figures/moral_llama.yaml
if [ "$CKPTS" = "last" ]; then
    MANIFEST=slurm_logs/moral_resolved.yaml
    python scripts/resolve_ckpts.py configs/figures/moral_llama.yaml "$MANIFEST"
fi

# The four figures are independent, so one failure must not skip the rest:
# plot_consistency_curves raises if a panel has <2 eval snapshots, and a Gemini
# quota error can take out any of the others.
failed=""
for fig in plot_consistency_curves plot_safety_sim plot_cf plot_capabilities; do
    echo "== [$fig] $(date)"
    python -m "evals.moral.$fig" --manifest "$MANIFEST" || failed="$failed $fig"
done

python -m evals.moral.plot_cf --manifest "$MANIFEST" --status || true

if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
