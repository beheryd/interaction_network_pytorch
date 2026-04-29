#!/bin/bash
# M5 full sweep — 40 runs (4 loss variants × 2 hidden × 5 seeds).
# Each task reads its (variant, config, hidden, seed, grad_clip_norm) tuple
# from scripts/sweep_matrix.csv at row $SLURM_ARRAY_TASK_ID and dispatches
# train.py with the right overrides. Run artifacts land under runs_m5/.
#
# Submit:  sbatch scripts/submit_m5_sweep.sbatch
# Status:  squeue -u $USER
# Cancel:  scancel <jobid>            (entire array)
#          scancel <jobid>_<taskid>   (single task)
#
#SBATCH --job-name=in_m5_sweep
#SBATCH --account=gershman_lab
#SBATCH --partition=shared
#SBATCH --output=runs_m5/slurm/%A_%a.out
#SBATCH --error=runs_m5/slurm/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --array=5-39%10
# Rows 0-4 (mov, h=10, seeds 0-4) are symlinked into runs_m5/ from
# runs_intercept_hp/ where they were already trained during the M5 prereq HP
# pass (clip=0.5, full 50K, all converged). 35 fresh tasks remain.

set -euo pipefail

cd /n/netscratch/kempner_gershman_lab/Lab/davidb/interaction_network_pytorch
source .venv/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p runs_m5/slurm

CSV=scripts/sweep_matrix.csv
ROW=$(awk -F, -v id="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==id {print; exit}' "$CSV")
if [ -z "$ROW" ]; then
    echo "ERROR: no row in $CSV for task_id=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# CSV columns: task_id,variant,config,hidden,seed,grad_clip_norm
IFS=, read -r task_id variant config hidden seed grad_clip_norm <<< "$ROW"

echo "=== task_id=$task_id ($variant h=$hidden s=$seed clip=$grad_clip_norm) on $(hostname) at $(date) ==="
python -m dmfc.training.train \
    --config "$config" \
    --seed "$seed" \
    --effect-dim "$hidden" \
    --grad-clip-norm "$grad_clip_norm" \
    --runs-root runs_m5
echo "=== done at $(date) ==="
