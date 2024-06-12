#!/bin/bash

# Script to automate the gridemissions data updates on an EC2 VM
# Scheduled hourly on cron with the next line
# @hourly /home/ec2-user/code/gridemissions/src/gridemissions/scripts/update_live_dataset.sh


source /home/ec2-user/.bashrc
conda activate py38

# Run live data update for webapp
ge_update_live_dataset

echo "Starting rsync to viz VM"
# Update visualization on EC2
# Depends on the "gridemissions-api" host being defined in ~/.ssh/config
data_path=$(python -c "from gridemissions import config; print(config['DATA_PATH_LIVE'])")
data_path_remote=/data/ubuntu/live/
rsync -v ${data_path}/last_update.txt gridemissions-api:${data_path_remote}last_update.txt
rsync -v ${data_path}/live_raw.csv gridemissions-api:${data_path_remote}live_raw.csv
rsync -v ${data_path}/live_elec.csv gridemissions-api:${data_path_remote}live_elec.csv
rsync -v ${data_path}/live_co2.csv gridemissions-api:${data_path_remote}live_co2.csv
rsync -rv --delete ${data_path}/d3map/ gridemissions-api:${data_path_remote}d3map

# Ping server so it reloads the data
curl https://api.gridemissions.com/info
