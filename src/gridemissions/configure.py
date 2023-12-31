import json
import logging.config
import os
from pathlib import Path
import sys
from typing import Union

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_environ_variable(
    expected_variable_name: str, is_path: bool = False
) -> Union[Path, str, None]:
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
            return Path(value)

    return value


CONFIG_DIR_PATH = get_environ_variable(
    "GRIDEMISSIONS_CONFIG_DIR_PATH", is_path=True
) or Path.home().joinpath((".config/gridemissions"))
CONFIG_DIR_PATH.mkdir(exist_ok=True, parents=True)

LOG_CONFIG_FILE_PATH = get_environ_variable(
    "GRIDEMISSIONS_LOG_CONFIG_FILE_PATH", is_path=True
) or CONFIG_DIR_PATH.joinpath("logging.conf")
CONFIG_FILE_PATH = get_environ_variable(
    "GRIDEMISSIONS_CONFIG_FILE_PATH", is_path=True
) or CONFIG_DIR_PATH.joinpath("config.json")

try:
    with open(CONFIG_FILE_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Generating config file in {CONFIG_DIR_PATH} with default values")
    data_dir_path = get_environ_variable(
        "GRIDEMISSIONS_DATA_DIR_PATH", is_path=True
    ) or Path.home().joinpath("data/gridemissions")
    tmp_dir_path = get_environ_variable(
        "GRIDEMISSIONS_TMP_DIR_PATH", is_path=True
    ) or Path.home().joinpath("tmp/gridemissions")
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

if "ENV" not in config:
    config["ENV"] = "PROD"

config["API_URL"] = "http://localhost:5000"

if config["ENV"] == "PROD":
    config["API_URL"] = "https://api.gridemissions.com"
config["S3_URL"] = "https://gridemissions.s3.us-east-2.amazonaws.com/"


def configure_logging(level="WARNING"):
    """
    Should be called by the user to configure logging

    Console logging is handled by the root logger's handler.
    To set the log level of the console to `console_log_level`, use
    ```
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(console_log_level)
    ```
    """
    logging.basicConfig(format=LOG_FORMAT, level=level)
    logger = logging.getLogger()
    logger.setLevel(level)


if __name__ == "__main__":
    try:
        ret_val = config.get(sys.argv[1]) or ""
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        ret_val = ""

    sys.stdout.write("%s\n" % ret_val)
