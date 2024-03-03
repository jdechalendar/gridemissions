.. _config:

Configuration
=============
Some configuration is needed for this project, to hold environment variables like data paths, API keys and passwords. The recommended option is to use a configuration file (``config.json``). A default is created for you the first time you import the package, in the folder ``~/.config/gridemissions``. ``~`` stands for your home directory. On Linux, this is what ``$HOME`` evaluates to, on Windows/OSX, this is the directory that contains your Documents, Desktop and Downloads folders.

Alternatively, configuration settings can be read from environment variables, e.g. with

.. code-block:: bash

 export GRIDEMISSIONS_CONFIG_DIR_PATH="$HOME/.config/gridemissions_test"

The config.json file
--------------------
Whenever you import the gridemissions module, the key-value pairs stored in ``config.json`` are loaded into a dictionary that then becomes available to you as the dictionary gridemissions.config. These can also be modified at runtime. At a minimum, your ``config.json`` should contain:

* ``DATA_PATH``: path to local data store, by default ``~/data/gridemissions``
* ``TMP_PATH``: for scratch data (e.g. when downloading data), by default ``~/tmp/gridemissions``

Supported Environment Variables
-------------------------------

.. code-block:: bash

 GRIDEMISSIONS_CONFIG_DIR_PATH       # the configuration directory (default: "$HOME/.config/gridemissions")
 GRIDEMISSIONS_LOG_CONFIG_FILE_PATH  # the file used to store logging (default: "$HOME/.config/gridemissions/logging.conf")
 GRIDEMISSIONS_CONFIG_FILE_PATH:     # the configuration file (default: "$HOME/.config/gridemissions/config.json")
 GRIDEMISSIONS_DEFAULT_LOGGING_CONF  # the default logging configuration (default can be read within ./src/gridemissions/configure.py")
 GRIDEMISSIONS_DATA_DIR_PATH         # the data directory (default: "$HOME/data/gridemissions")
 GRIDEMISSIONS_TMP_DIR_PATH          # the temporary data directory (default: "$HOME/tmp/gridemissions")
