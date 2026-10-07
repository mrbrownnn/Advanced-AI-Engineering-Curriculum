import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
sys.path.insert(0, str(LAB / "fixtures"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
