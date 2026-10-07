"""Select the implementation under test: starter/ (default) or the reference (LAB_IMPL=solution)."""
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
MODULE = LAB.parents[1].name
if os.environ.get("LAB_IMPL") == "solution":
    IMPL = LAB.parents[2] / "labs-solutions" / MODULE / LAB.name
else:
    IMPL = LAB / "starter"
sys.path.insert(0, str(IMPL))
