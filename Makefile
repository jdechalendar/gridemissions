# Avoids DeprecationWarning
# See https://github.com/jupyter/jupyter_core/blob/main/jupyter_core/paths.py#L208
export JUPYTER_PLATFORM_DIRS = 1

snapshot: ## (over)write `syrupy` snapshots
	pytest --snapshot-update test/

test: ## Run `pytest`, `coverage html`, and `flake8`
	pytest --cov=src --snapshot-details --snapshot-warn-unused test/
	flake8

test_fast: ## Run `flake8` and `SKIP_SLOW_TESTS=true pytest`
	SKIP_SLOW_TESTS=true pytest --snapshot-warn-unused test/
	flake8
