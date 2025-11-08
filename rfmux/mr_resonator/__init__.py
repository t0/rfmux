"""
This package is derived from and based on the mr_resonator project by Maclean Rouble:
https://github.com/macleaner/mr_resonator

Contents (trimmed for Mock CRS):
- mr_complex_resonator: High-level resonator physics with quasiparticle modeling
- mr_lekid: Circuit model for lumped element KIDs
- jit_physics: JIT-compiled physics functions for performance

Modifications in this repository include:
- Trimmed to include only functions used by the Mock CRS framework
- Removal of plotting and SciPy dependencies
- Consolidation and integration with local JIT physics approximations

Original project license: see rfmux/mr_resonator/LICENSE (upstream LICENSE retained)
"""

# Import main classes for convenient access
from .mr_complex_resonator import MR_complex_resonator
from .mr_lekid import MR_LEKID
from . import jit_physics

__all__ = ['MR_complex_resonator', 'MR_LEKID', 'jit_physics']
