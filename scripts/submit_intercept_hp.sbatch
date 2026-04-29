#!/bin/bash
# 5-seed × 50K-step Intercept (mov) HP confirmation. Each array task trains
# one seed end-to-end with the configs/in_intercept.yaml settings (lr=1e-3,
# grad_clip_norm=0.5, weight_decay=0.0). Per-seed run dir lands under
# runs_intercept_hp/in_mov_h10_s<seed>_<ts>_<git>/.
#
# Submit:  sbatch scripts/submit_intercept_hp.sbatch
# Status:  squeue -u $USER
# Cancel:  scancel <jobid>
#
#SBATCH --job-name=in_intercept_hp
#SBATCH --account=gershman_lab
#SBATCH --partition=shared
#SBATCH --output=runs_intercept_hp/slurm/%A_%a.out
#SBATCH --error=runs_intercept_hp/slurm/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --array=0-4

set -euo pipefail

cd /n/netscratch/kempner_gershman_lab/Lab/davidb/interaction_network_pytorch
source .venv/bin/activate

# torch on CPU: keep threading modest so 5 parallel tasks share the node nicely.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p runs_intercept_hp/slurm

echo "=== task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
python -m dmfc.training.train \
    --config configs/in_intercept.yaml \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --runs-root runs_intercept_hp
echo "=== done at $(date) ==="
