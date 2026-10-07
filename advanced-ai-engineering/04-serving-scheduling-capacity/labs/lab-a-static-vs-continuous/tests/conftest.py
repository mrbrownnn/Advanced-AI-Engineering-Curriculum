import pathlib, sys
LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labkit
labkit.use_impl(LAB)
