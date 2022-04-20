import json
import logging.config
import os
from pathlib import Path
from typing import Union


def get_environ_variable(expected_variable_name: str, is_path: bool = False) -> Union[Path, str, None]:
    """
    Get an environment variable.

    Optionally retrieves the variable as a path.
    Parameters
    ----------
    expected_variable_name: The expected environment variable name.
    is_path: Whether to try and interpret the variable as a path.

    Returns
    -------
    value: Union[Path, str, None]
        The value of the environment variable.
    """
    value = os.environ.get(expected_variable_name)

    if value:
        if is_path:
            try:
                return Path(value)
            except:
                return None

    return value


CONFIG_DIR_PATH = get_environ_variable("GRIDEMISSIONS_CONFIG_DIR_PATH", is_path=True) or Path.home().joinpath(
    (".config/gridemissions"))
CONFIG_DIR_PATH.mkdir(exist_ok=True, parents=True)

LOG_CONFIG_FILE_PATH = get_environ_variable("GRIDEMISSIONS_LOG_CONFIG_FILE_PATH",
                                            is_path=True) or CONFIG_DIR_PATH.joinpath(
    "logging.conf")
CONFIG_FILE_PATH = get_environ_variable("GRIDEMISSIONS_CONFIG_FILE_PATH", is_path=True) or CONFIG_DIR_PATH.joinpath(
    "config.json")

DEFAULT_LOGGING_CONF = get_environ_variable("GRIDEMISSIONS_DEFAULT_LOGGING_CONF") or """# Config file for logging
[loggers]
keys=root,scraper,gridemissions

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_scraper]
level=INFO
handlers=consoleHandler
qualname=scraper
propagate=0

[logger_gridemissions]
level=INFO
handlers=consoleHandler
qualname=gridemissions
propagate=0

[handler_consoleHandler]
class=StreamHandler
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=
"""

try:
    with open(CONFIG_FILE_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Generating config file in {CONFIG_DIR_PATH} with default values")
    data_dir_path = get_environ_variable("GRIDEMISSIONS_DATA_DIR_PATH", is_path=True) or Path.home().joinpath(
        "data/gridemissions")
    tmp_dir_path = get_environ_variable("GRIDEMISSIONS_TMP_DIR_PATH", is_path=True) or Path.home().joinpath(
        "tmp/gridemissions")
    data_dir_path.mkdir(exist_ok=True, parents=True)
    tmp_dir_path.mkdir(exist_ok=True, parents=True)
    config = {
        "DATA_PATH": str(data_dir_path),
        "TMP_PATH": str(tmp_dir_path),
    }

    # Store for later use
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f, indent=4)

    # Sanity check: reload
    with open(CONFIG_FILE_PATH, "r") as f:
        config = json.load(f)

for k in config:
    if k.endswith("_PATH"):  # by convention, this is a directory
        config[k] = Path(config[k])

# Setup logging
try:
    logging.config.fileConfig(LOG_CONFIG_FILE_PATH)
except KeyError:
    print(f"Generating logging config file in {CONFIG_DIR_PATH} with default values")
    with open(LOG_CONFIG_FILE_PATH, "w") as f:
        f.write(DEFAULT_LOGGING_CONF)
    logging.config.fileConfig(LOG_CONFIG_FILE_PATH)

if "ENV" not in config:
    config["ENV"] = ""
