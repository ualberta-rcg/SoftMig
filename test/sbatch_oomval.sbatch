#!/bin/bash
#SBATCH --reservation=softmig
#SBATCH --gres=gpu:l40s.2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --job-name=oomval
#SBATCH --output=%x_%j.out

# Run our full validation suite from inside an sbatch job to confirm the
# native OOM behavior + legacy killer opt-in both work under batch.
module load cuda/12.6

SOFTMIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$SOFTMIG_ROOT/test/results/sbatch_oomval_${SLURM_JOB_ID}"
mkdir -p "$OUT"
SOFTMIG_ROOT="$SOFTMIG_ROOT" OUT="$OUT" bash "$SOFTMIG_ROOT/test/run_oom_validation.sh"
