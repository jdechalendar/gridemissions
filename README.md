# gridemissions: Tools for power sector emissions tracking
<img src="https://user-images.githubusercontent.com/20404131/129465144-5b086d9b-6c46-462f-a036-3f1e4cd958eb.png" width="50%" align="right">

The tools in this repository power the visualization at [energy.stanford.edu/gridemissions](https://energy.stanford.edu/gridemissions), updated hourly. Associated datasets on electricity and emissions are made publicly available. In addition to tools to create the data, the `gridemissions` package provides a module to retrieve data from the API and methods to load and manipulate the data. This README file serves as a quick primer for the tools in this repository and includes a short FAQ, please refer to the documentation for more.

## Documentation
Documentation generated using Sphinx can be found [here](https://docs.gridemissions.jdechalendar.su.domains/).

## README Contents
* [How the datasets are created](https://github.com/jdechalendar/gridemissions#how-the-datasets-are-created)
* [Retrieving data from the API](https://github.com/jdechalendar/gridemissions#Retrieving-data-from-the-API)
* [FAQ](https://github.com/jdechalendar/gridemissions#FAQ)
* [Installation](https://github.com/jdechalendar/gridemissions#Installation)
* [The `GraphData` class](https://github.com/jdechalendar/gridemissions#The-GraphData-class)


## How the datasets are created
Two main operations are needed to create the datasets for the the visualization at [energy.stanford.edu/gridemissions](https://energy.stanford.edu/gridemissions).

### 1. Consumption-based emissions
Electric grid data on production, consumption and exchanges, along with the emissions associated with electricity production, are used to compute the emissions embodied in electricity **consumption**. By default, we are using IPCC Life-Cycle Assessment emissions factors to compute the emissions associated with generating electricity from different sources, so the CO2 data we release are in units of CO2-eq. If you wish to use different emissions factors, or factors for other quantities (e.g. SO2, NOx, PM2.5, or H2O), you can use the tools in this package to generate corresponding consumption-based data. A tutorial on how to do this will be available soon.

For more on this operation, see "Tracking emissions in the US electricity system", by Jacques A. de Chalendar, John Taggart and Sally M. Benson. Proceedings of the National Academy of Sciences Dec 2019, 116 (51) 25497-25502; DOI: 10.1073/pnas.1912950116

### 2. Physics-based data reconciliation
Raw electric grid data typically have errors and inconsistencies, but we need "clean" data to compute consumption-based emissions. We use an optimization-based algorithm to reconcile the raw data while enforcing certain physical constraints, e.g. conservation of energy. We publish both the raw and reconciled electric data that we use.

For more on this operation, see "Physics-informed data reconciliation framework for real-time electricity and emissions tracking", by Jacques A. de Chalendar and Sally M. Benson. Applied Energy Dec 2021; DOI: 10.1016/j.apenergy.2021.117761 [ArXiv preprint](https://arxiv.org/abs/2103.05663).

## Bulk dataset
The main source of data for this project is the [US EIA's Grid Monitor](https://www.eia.gov/electricity/gridmonitor). Bulk files from that source are now available in six-month chunks, so we have switched to doing the same. The processed files can be downloaded all at once from [here](https://gridemissions.s3.us-east-2.amazonaws.com/processed.tar.gz) (1.3 GB as of Jan 16, 2024). See notebook `notebooks/Bulk Data Demo.ipynb` that makes a few plots from this dataset as a sanity check. We plan to also host a version of these datasets converted to Parquet for easier access -- this README will be updated when that is done.

## Retrieving data from the API
For a quick introduction to the package, see the notebooks in the `notebooks/` folder. The [API Demo.ipynb](https://colab.research.google.com/drive/1HYHqiA2iA-vVMuqFHrKtUUkdPLN5UJYS) notebook can also be loaded on Colab and shows how data can be retrieved from the API and then manipulated using the `GraphData` methods. Note that only one month of historical data is available from the API.

A download script is provided to quickly download data from the API and can be used after installing the package (see below):
```bash
# Downloads data for CAISO for a year
gridemissions_download --variable=co2 --region=CISO --start=20190101 --end=20200101

# Print help
gridemissions_download -h

# Download one of the datasets in bulk
gridemissions_download --variable co2 --all
```
Note that the data are downloaded to the `DATA_PATH` you configured during setup (see Installation notes below).

You can also use the `api` module to retrieve data programmatically. This is what the `gridemissions_download` script uses under the hood.
```python
from gridemissions import api

# Download CO2 emissions embodied in electricity consumption in the California ISO (CISO)
data = api.retrieve(dataset="co2", region="CISO", field="D")

# Download electricity generated by the Electric reliability council of Texas (ERCO) and in the Bonneville Power Administration (BPAT)
data = api.retrieve(dataset="elec", region=["ERCO", "BPAT"], field="NG")
```
By default, the `api.retrieve` function returns data in a pandas DataFrame. We also provide an abstraction to load and manipulate data called `GraphData`. This object is a light wrapped around a pandas DataFrame that provides convenient functionality to represent and access data from different fields on a graph. More on that below.

### Data naming conventions
In the datasets that is generated from this work, we use the following conventions for naming columns (see `eia_api.py`). Replace `%s` in the following dictionaries by the balancing acronyms listed [here](https://www.eia.gov/electricity/gridmonitor/about).
* For electricity, we follow the naming convention followed by the US EIA data source we are using:
```python
"E": {
      "D": "EBA.%s-ALL.D.H",  # Demand (Consumption)
      "NG": "EBA.%s-ALL.NG.H",  # Generation
      "TI": "EBA.%s-ALL.TI.H",  # Total Interchange
      "ID": "EBA.%s-%s.ID.H",  # Interchange
  }
```
For example, `"EBA.CISO-ALL.D.H"` is the column for demand in the California ISO.
* For all other variables, we use a different convention, that only uses underscores as separators. For example, for carbon dioxide:
```python
"CO2": {
    "D": "CO2_%s_D",  # Demand (Consumption)
    "NG": "CO2_%s_NG",  # Generation
    "TI": "CO2_%s_TI",  # Total Interchange
    "ID": "CO2_%s-%s_ID",  # Interchange
}
```
For example, `"CO2_CISO_D"` is the column for consumed emissions in the California ISO, `"CO2_CISO_NG"` is the column for produced emissions in the California ISO.

## FAQ
### I tried to retrieve data from the API from 2018 without success
The backend API only stores a month's worth of data (to save on AWS costs). We suggest downloading data in bulk instead.

### Where do the emissions factors come from?
These are life-cycle emissions factors from the [IPCC](https://www.ipcc.ch/report/renewable-energy-sources-and-climate-change-mitigation/) (Table A.II.4 on page 982 of the report at the link). If you want, you can use other emissions factors. [This](https://github.com/jdechalendar/gridemissions/blob/main/src/gridemissions/emissions.py#L14-L28) is where they are being read in by the codebase. If you pass in custom emissions factors, you can then re-run the code to generate estimates using your favorite ones. It would also not be too difficult to modify this code to make the emissions factors depend on the balancing-area and time of year, although that would require a bit more work.

## Installation
Clone this repository on your machine using HTTPS:
```
git clone https://github.com/jdechalendar/gridemissions.git
```
or using SSH (your GitHub account needs to have been configured in this case):
```
git clone git@github.com:jdechalendar/gridemissions.git
```
From the `gridemissions` directory (the one that contains this README file), install this repository:
```
pip install .
```
Installing the project in this way means that you can now use statements like `import gridemissions` to make the code in this repository accessible anywhere on your system.
To install the optional dependencies as well (needed if you would like to run the automated data cleaning workflows)
```
pip install .[all]
```
If you intend to modify the code, you may want to install with the editable flag:
```
pip install -e .[all]
```
As explained [here](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs), this installs the package in setuptools' ["Development mode"](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html) so that you don't need to re-build the project every time you edit the code.
Open a Python interpreter and import the package to create the default configuration files for the project. When you import the package, it will check if configuration files exist. If not, a message will be printed to the screen to tell you where the configuration files are created on your system.
Optionally, you can customize the configuration files. See the configuration section for details.

### Configuration
Some configuration is needed for this project, to hold environment variables like data paths, API keys and passwords. The recommended option is to use a configuration file (`config.json`). A default is created for you the first time you import the package, in the folder `~/.config/gridemissions`. `~` stands for your home directory. On Linux, this is what `$HOME` evaluates to, on Windows/OSX, this is the directory that contains your Documents, Desktop and Downloads folders.

Alternatively, configuration settings can be read from environment variables, e.g. with
```bash
export GRIDEMISSIONS_CONFIG_DIR_PATH="$HOME/.config/gridemissions_test"
```

#### The config.json file
Whenever you import the `gridemissions` module, the key-value pairs stored in `config.json` are loaded into a dictionary that then becomes available to you as the dictionary `gridemissions.config`. These can also be modified at runtime. At a minimum, your `config.json` should contain:
* `DATA_PATH`: path to local data store, by default `~/data/gridemissions`
* `TMP_PATH`: for scratch data (e.g. when downloading data), by default `~/tmp/gridemissions`

#### Supported Environment Variables

```text
GRIDEMISSIONS_CONFIG_DIR_PATH:      the configuration directory (default: "$HOME/.config/gridemissions")
GRIDEMISSIONS_LOG_CONFIG_FILE_PATH: the file used to store logging (default: "$HOME/.config/gridemissions/logging.conf")
GRIDEMISSIONS_CONFIG_FILE_PATH:     the configuration file (default: "$HOME/.config/gridemissions/config.json")
GRIDEMISSIONS_DEFAULT_LOGGING_CONF: the default logging configuration (default can be read within ./src/gridemissions/configure.py")
GRIDEMISSIONS_DATA_DIR_PATH:        the data directory (default: "$HOME/data/gridemissions")
GRIDEMISSIONS_TMP_DIR_PATH:         the temporary data directory (default: "$HOME/tmp/gridemissions")
```
