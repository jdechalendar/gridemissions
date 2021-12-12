import pathlib
from os.path import expanduser
import json
import logging.config


CONF_DIR = pathlib.Path(expanduser("~/.config/gridemissions"))
CONF_DIR.mkdir(exist_ok=True)

LOG_CONFIG_FILE_PATH = CONF_DIR / "logging.conf"
CONFIG_FILE_PATH = CONF_DIR / "config.json"

DEFAULT_LOGGING_CONF = """# Config file for logging
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
    print(f"Generating config file in {CONF_DIR} with default values")
    default_data_dir = pathlib.Path(expanduser("~/data/gridemissions"))
    default_tmp_dir = pathlib.Path(expanduser("~/tmp/gridemissions"))
    default_data_dir.mkdir(exist_ok=True, parents=True)
    default_tmp_dir.mkdir(exist_ok=True, parents=True)
    config = {
        "DATA_PATH": str(default_data_dir),
        "TMP_PATH": str(default_tmp_dir),
    }

    # Store for later use
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f, indent=4)

    # Sanity check: reload
    with open(CONFIG_FILE_PATH, "r") as f:
        config = json.load(f)

for k in config:
    if k.endswith("_PATH"):  # by convention, this is a directory
        config[k] = pathlib.Path(config[k])

# Setup logging
try:
    logging.config.fileConfig(LOG_CONFIG_FILE_PATH)
except KeyError:
    print(f"Generating logging config file in {CONF_DIR} with default values")
    with open(LOG_CONFIG_FILE_PATH, "w") as f:
        f.write(DEFAULT_LOGGING_CONF)
    logging.config.fileConfig(LOG_CONFIG_FILE_PATH)

if "ENV" not in config:
    config["ENV"] = ""
