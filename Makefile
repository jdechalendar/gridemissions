# Avoids DeprecationWarning
# See https://github.com/jupyter/jupyter_core/blob/main/jupyter_core/paths.py#L208
export JUPYTER_PLATFORM_DIRS = 1

data_path := $(shell python src/gridemissions/configure.py DATA_PATH)
fig_path := $(shell python src/gridemissions/configure.py FIG_PATH)

.PHONY: install snapshot test test_fast bulk bulk_upload bulk_report

install: ## install all requirements in editable mode
	pip install -e '.[all]'

snapshot: ## (over)write `syrupy` snapshots
	pytest --snapshot-update test/

test: ## Run `pytest`, `coverage html`, and `flake8`
	pytest --cov=src --snapshot-details --snapshot-warn-unused test/
	flake8

test_fast: ## Run `flake8` and `SKIP_SLOW_TESTS=true pytest`
	SKIP_SLOW_TESTS=true pytest --snapshot-warn-unused test/
	flake8

# Note 1: Files are only downloaded by wget if they do not exist (with wget -nc).
# Note 2: Files are only processed if the "co2" file that is created at the end of the
# workflow does not exist. Delete files to force an update
bulk:
		DATA_PATH=${data_path} bash src/gridemissions/scripts/bulk_download_grid_monitor.sh
		python src/gridemissions/scripts/bulk_process.py
		cd ${data_path}/EIA_Grid_Monitor && tar -czf processed.tar.gz processed/

# Note: using --sse AES256 is required for the Stanford AWS account
bulk_upload:  ## Upload bulk dataset to s3 bucket
	aws s3 sync --sse AES256 --delete --exclude "*" --include="processed.tar.gz" ${data_path}/EIA_Grid_Monitor/ s3://gridemissions/

bulk_report: ## Create heatmap and timeseries reports and upload to S3
	ge_report --report heatmap
	ge_report --report timeseries
	aws s3 sync --sse AES256 --delete --exclude "*" --include "*.pdf"  --include "*.png" --include "*.json" ${fig_path} s3://gridemissions/figures
