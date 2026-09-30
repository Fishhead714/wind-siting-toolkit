"""wind_layout: terrain-aware turbine layout screening pipeline.

Screening / pre-feasibility tool. Not a bankable energy yield assessment;
see ``wind_layout.pipeline.OUTPUT_DISCLAIMER``.
"""
from . import rasterutils, spacingcheck, terrainclassify, terrainlayout, siteconditions  # noqa: F401

__version__ = "0.1.0"

#: The pipeline and optimiser modules import TOPFARM / PyWake at import time, so
#: they are loaded on demand: ``from wind_layout import pipeline``.
