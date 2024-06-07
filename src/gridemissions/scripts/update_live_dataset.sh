#!/bin/bash

# Script to automate the gridemissions data updates on an EC2 VM
# Scheduled hourly on cron with the next line
# @hourly /home/ec2-user/code/gridemissions/src/gridemissions/scripts/update_live_dataset.sh


source /home/ec2-user/.bashrc
conda activate py38

# Run data update for webapp
ge_update

data_path=/data/ec2-user/analysis/


echo "Starting s3 sync"
aws s3 sync --sse AES256 --delete --exclude "*" --include "*.tar.gz" ${data_path}s3/ s3://gridemissions


data_path_remote=/data/ubuntu/analysis/

echo "Starting rsync to viz VM"
# Update visualization on EC2
# Depends on the "gridemissions-api" host being defined in ~/.ssh/config
rsync -v ${data_path}webapp/last_update.txt gridemissions-api:${data_path_remote}webapp/last_update.txt
rsync -v ${data_path}webapp/data_extract/EBA_raw.csv gridemissions-api:${data_path_remote}webapp/EBA_raw.csv 
rsync -v ${data_path}webapp/data_extract/EBA_elec.csv gridemissions-api:${data_path_remote}webapp/EBA_elec.csv 
rsync -v ${data_path}webapp/data_extract/EBA_co2.csv gridemissions-api:${data_path_remote}webapp/EBA_co2.csv 
rsync -rv --delete ${data_path}webapp/d3map/ gridemissions-api:${data_path_remote}webapp/d3map

# Ping server so it reloads the data
curl https://api.gridemissions.com/info
