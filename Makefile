# Avoids DeprecationWarning
# See https://github.com/jupyter/jupyter_core/blob/main/jupyter_core/paths.py#L208
export JUPYTER_PLATFORM_DIRS = 1

data_path := $(shell python src/gridemissions/configure.py DATA_PATH)

.PHONY: iea_data clean install snapshot test test_fast

iea_data: ${data_path}/EIA_Grid_Monitor/downloads/.dummy_file_for_make

# Note: the rules below do not check if there are new files to download/process
# Could be improved to only download/process certain files
${data_path}/EIA_Grid_Monitor/downloads/.dummy_file_for_make:
		DATA_PATH=${data_path} bash src/gridemissions/scripts/bulk_download_grid_monitor.sh
		touch ${data_path}/EIA_Grid_Monitor/.dummy_file_for_make

bulk: ${data_path}/EIA_Grid_Monitor/downloads/.dummy_file_for_make
		python src/gridemissions/scripts/bulk_process.py

clean:
		rm ${data_path}/EIA_Grid_Monitor/.dummy_file_for_make

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
