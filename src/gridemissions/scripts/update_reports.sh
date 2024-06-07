#!/bin/bash

# Script to automate the gridemissions data updates on the VM
# Scheduled weekly on cron with the next line
# @weekly /home/ec2-user/code/gridemissions/src/gridemissions/scripts/update_report.sh


source /home/ec2-user/.bashrc
conda activate py38

ge_report --report heatmap
ge_report --report timeseries

fig_path=(python src/gridemissions/configure.py FIG_PATH)
aws s3 sync --sse AES256 --delete --exclude "*" --include "heatmap_report/*.pdf" ${fig_path} s3://gridemissions
aws s3 sync --sse AES256 --delete --exclude "*" --include "heatmap_report/*.png" ${fig_path} s3://gridemissions
aws s3 sync --sse AES256 --delete --exclude "*" --include "heatmap_report/*.json" ${fig_path} s3://gridemissions
