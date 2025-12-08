#!/bin/bash
# SLURM epilog.sh for softmig
# This script runs after each job and cleans up softmig config files

# ===== softmig CLEANUP =====
# Delete config file for this job (epilog script is responsible for cleanup)
if [[ ! -z "$SLURM_JOB_ID" ]]; then
    CONFIG_FILE="/var/run/softmig/${SLURM_JOB_ID}.conf"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        rm -f "$CONFIG_FILE"
        logger -t slurm_epilog "Job $SLURM_JOB_ID: Cleaned up softmig config file $CONFIG_FILE"
    fi
fi

