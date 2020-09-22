# gridemissions
Tools for power sector emissions tracking

# Installation
Clone this repository and then run `pip install .` from within it.

# Usage
See the docs for more advanced usage. A download script is provided to quickly download data:
```
# Download data for CAISO for a year
download_emissions --variable=co2 --ba=CISO --start=20190101 --end=20200101

# Print help
download_emissions -h
```