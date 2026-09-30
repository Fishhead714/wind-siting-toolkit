"""Turbine-class suitability screening: does a turbine's design envelope cover a site?

Screening level only; see ``engine.DISCLAIMER``. Not an energy-yield tool.
"""
from .engine import DISCLAIMER, ScreeningConfig, SuitabilityEngine
from .evidence import decide_tier, wording
from .iec import ClassTable, default_table
from .report import render_markdown
from .site import SiteConditions
from .turbine import TurbineSpec

__all__ = ["DISCLAIMER", "ScreeningConfig", "SuitabilityEngine", "decide_tier", "wording",
           "ClassTable", "default_table", "render_markdown", "SiteConditions", "TurbineSpec"]
__version__ = "0.1.0"
