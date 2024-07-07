import importlib.metadata

from .configure import config, configure_logging
from .load import GraphData, read_csv, load_bulk
from .emissions import EmissionsCalc
import eia_api_v1
import eia_api_v2

try:
    from .clean import BasicCleaner, RollingCleaner, CvxCleaner

    has_optional_dependencies = True

except ModuleNotFoundError:  # Cleaning depends on pptional dependencies
    has_optional_dependencies = False

__all__ = [
    "config",
    "configure_logging",
    "eia_api_v1",
    "eia_api_v2",
    "EmissionsCalc",
    "GraphData",
    "read_csv",
    "load_bulk",
]
__version__ = importlib.metadata.version("gridemissions")

if has_optional_dependencies:
    __all__ += ["BasicCleaner", "CvxCleaner", "RollingCleaner"]
