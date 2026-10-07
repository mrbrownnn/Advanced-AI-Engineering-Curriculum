"""Select the implementation under test.

default            -> ../starter
LAB_IMPL=solution  -> labs-solutions/<module>/<lab>   (instructor tree)
LAB_IMPL=bugged    -> starter/bugged_<name>.py imported as <name>   (BREAK)
"""
import importlib
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
MODULE = LAB.parents[1]
IMPL = os.environ.get("LAB_IMPL", "starter")

if IMPL == "solution":
    path = MODULE.parent / "labs-solutions" / MODULE.name / LAB.name
else:
    path = LAB / "starter"
sys.path.insert(0, str(path))

if IMPL == "bugged":
    for f in (LAB / "starter").glob("bugged_*.py"):
        sys.modules[f.stem[len("bugged_"):]] = importlib.import_module(f.stem)
