# coding=utf-8
"""Loader for ``data/pipeline.json`` -- the single configuration file.

Nothing is configured *here*; this module only reads the JSON and exposes it,
both as the whole document (``CFG``) and as the storage names the
DataController uses. Edit ``pipeline.json``, never this file.
"""

from __future__ import annotations

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pipeline.json")

with open(CONFIG_PATH, encoding="utf-8") as _f:
    CFG = json.load(_f)

# ---- storage (used by DataController) -------------------------------------
_S = CFG["storage"]
DATA_ROOT = _S["root"]
INVENTORY = _S["inventory"]
ORBITS_DIR = _S["orbits"]
ORBITS_STAGING = _S["orbits_staging"]
RESCORE = _S["rescore"]
TOKENISED = _S["tokenised"]
DATASETS = _S["datasets"]

# ---- the rest of the document ---------------------------------------------
HPC = CFG["hpc"]
STAGES = CFG["stages"]


def stage(name: str) -> dict:
    """Configuration block for one pipeline stage."""
    return STAGES[name]
