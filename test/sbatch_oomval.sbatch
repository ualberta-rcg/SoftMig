#!/bin/bash
#SBATCH --reservation=softmig
#SBATCH --gres=gpu:l40s.2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --job-name=oomval
#SBATCH --output=/scratch/rahimk/SoftMig/test/results/sbatch_oomval_%j.out

# Run our full validation suite from inside an sbatch job to confirm the
# native OOM behavior + legacy killer opt-in both work under batch.
module load cuda/12.6

OUT="/scratch/rahimk/SoftMig/test/results/sbatch_oomval_${SLURM_JOB_ID}"
mkdir -p "$OUT"
SOFTMIG_ROOT=/scratch/rahimk/SoftMig OUT="$OUT" bash /scratch/rahimk/SoftMig/test/run_oom_validation.sh
