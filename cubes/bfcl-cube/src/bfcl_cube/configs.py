"""Canonical bfcl-cube benchmark configs.

from bfcl_cube import BFCL_CONFIGS
benchmark = BFCL_CONFIGS["default"]      # all single-turn categories
benchmark = BFCL_CONFIGS["non_live"]     # NON_LIVE only
benchmark = BFCL_CONFIGS["live"]         # LIVE only
"""

from cube.core import ConfigRegistry

from bfcl_cube.benchmark import BfclBenchmarkConfig

BFCL_CONFIGS: ConfigRegistry[BfclBenchmarkConfig] = ConfigRegistry(
    {
        "default": BfclBenchmarkConfig(),
        "non_live": BfclBenchmarkConfig().named_subset("non_live"),
        "live": BfclBenchmarkConfig().named_subset("live"),
    }
)
