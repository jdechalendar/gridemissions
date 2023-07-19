from .configure import config, configure_logging
from .load import GraphData, read_csv
from .emissions import EmissionsCalc

try:
    from .clean_v2 import BasicCleaner, RollingCleaner, CvxCleaner

    has_optional_dependencies = True

except ModuleNotFoundError:  # Cleaning depends on pptional dependencies
    has_optional_dependencies = False

__all__ = ["config", "configure_logging", "EmissionsCalc", "GraphData", "read_csv"]

if has_optional_dependencies:
    __all__ += ["BasicCleaner", "CvxCleaner", "RollingCleaner"]
