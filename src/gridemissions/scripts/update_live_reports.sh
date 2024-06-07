#!/bin/bash

# Script to automate the gridemissions data updates on the VM
# Scheduled weekly on cron with the next line
# @weekly /home/ec2-user/code/gridemissions/src/gridemissions/scripts/update_live_report.sh


source /home/ec2-user/.bashrc
conda activate py38

ge_report --report heatmap --year 2021
ge_report --report timeseries

data_path=/data/ec2-user/analysis/
aws s3 sync --sse AES256 --delete --exclude "*" --include "figures/*.pdf" ${data_path}s3/ s3://gridemissions
aws s3 sync --sse AES256 --delete --exclude "*" --include "figures/*.png" ${data_path}s3/ s3://gridemissions
aws s3 sync --sse AES256 --delete --exclude "*" --include "figures/*.json" ${data_path}s3/ s3://gridemissions
