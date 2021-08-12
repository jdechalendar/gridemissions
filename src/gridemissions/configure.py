import os
from os.path import expanduser
import json
import logging.config

CONFIG_FILE_PATH = expanduser("~/.config/gridemissions/config.json")
LOG_CONFIG_FILE_PATH = expanduser("~/.config/gridemissions/logging.conf")
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
    print("Generating config file in ~/.config/gridemissions/ with default values")
    os.makedirs(expanduser("~/.config/gridemissions"), exist_ok=True)
    os.makedirs(expanduser("~/data"), exist_ok=True)
    config = {"DATA_PATH": expanduser("~/data"), "TMP_PATH": expanduser("~/tmp")}

    # Store for later use
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f)

try:
    logging.config.fileConfig(LOG_CONFIG_FILE_PATH)
except KeyError:
    print("Generating logging config file in ~/.config/gridemissions/ with default values")
    with open(LOG_CONFIG_FILE_PATH, "w") as f:
        f.write(DEFAULT_LOGGING_CONF)

if "ENV" not in config:
    config["ENV"] = ""
