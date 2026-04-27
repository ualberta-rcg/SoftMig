#!/bin/bash
# SLURM epilog.sh for softmig
# This script runs after each job and cleans up softmig config files

# ===== softmig CLEANUP =====
# Delete config file(s) for this job (handles both regular and array jobs)
if [[ -n "$SLURM_JOB_ID" ]]; then
    # Remove the base config and any array-suffixed configs (e.g. {jobid}_1.conf)
    removed=0
    for f in /var/run/softmig/${SLURM_JOB_ID}.conf /var/run/softmig/${SLURM_JOB_ID}_*.conf; do
        if [[ -f "$f" ]]; then
            rm -f "$f"
            removed=$((removed + 1))
        fi
    done
    if [[ $removed -gt 0 ]]; then
        logger -t slurm_epilog "Job $SLURM_JOB_ID: Cleaned up $removed softmig config file(s)"
    fi
fi

