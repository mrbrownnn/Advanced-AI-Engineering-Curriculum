import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labenv  # noqa: E402

labenv.select_impl(LAB)

import torch  # noqa: E402

torch.set_default_device("cpu")
torch.use_deterministic_algorithms(True)
