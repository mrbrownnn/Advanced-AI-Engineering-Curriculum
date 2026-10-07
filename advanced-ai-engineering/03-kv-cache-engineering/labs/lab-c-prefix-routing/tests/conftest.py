import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labkit  # noqa: E402

# Lab C reuses the Lab B block manager (yours, or the reference under LAB_IMPL=solution).
labkit.use_impl(LAB, "lab-b-paged-block-manager")
