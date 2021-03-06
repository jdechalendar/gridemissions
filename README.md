# gridemissions: Tools for power sector emissions tracking

The tools in this repository power the visualization [here](https://energy.stanford.edu/gridemissions), updated hourly. Data is made publicly available and hosted on an AWS S3 bucket. The notebooks [here](tocome.com) can be used to quickly explore the data that are available.

## References
* "Tracking emissions in the US electricity system", by Jacques A. de Chalendar, John Taggart and Sally M. Benson. Proceedings of the National Academy of Sciences Dec 2019, 116 (51) 25497-25502; DOI: 10.1073/pnas.1912950116
* "Physics-informed data reconciliation framework for real-time electricity and emissions tracking", by Jacques A. de Chalendar and Sally M. Benson. In review.

## Installation
Clone or download this repository from Github and then install:
* the base packages needed to make calls to the API: `pip install .`
* all packages (including those needed for automated data cleaning): `pip install .[all]`
* in editable mode to make it easier to modify the codebase: `pip install -e .[all]`

## Configuration
Two configuration files are generated automatically in a directory called  `~/.config/gridemissions` the first time you import the package:
* `config.json` stores environment variables
* logging.conf configures different loggers

You can then modify these files to change confirguration, such as the `DATA_PATH` variable in `config.json` which is where the package will look for data (by default `~/data`).

## Documentation
Documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) and was generated in the `docs` folder. Open the index.html file in a browser to view the documentation. To re-generate the documentation you shoud be able to simply do
```
cd doc
make html
```
This requires that you have `sphinx` installed.

## API Usage
A download script is provided to quickly download data:
```python
# Download data for CAISO for a year
download_emissions --variable=co2 --ba=CISO --start=20190101 --end=20200101

# Print help
download_emissions -h
```

You can also use the client library to retrieve data programmatically:
```python
from gridemissions import api

# Download data for CAISO for a year
data = api.retrieve(variable="co2", ba="CISO", start="20190101", end="20200101", field="D")

# Download electricity generation data for ERCOT and BPAT for a year:
data = api.retrieve(variable="elec", ba=["ERCOT", "BPAT"], start="20190101", end="20200101", field="NG")

import pandas as pd
from io import StringIO
print(pd.read_csv(StringIO(data)).head())
```
See the docs for more advanced usage.

## Physics-informed data cleaning
Instructions to come.
