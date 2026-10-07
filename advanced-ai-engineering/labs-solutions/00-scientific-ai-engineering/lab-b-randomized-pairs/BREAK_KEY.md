# LAB B BREAK Key

`starter/bugged_benchmark.py` ignores randomized assignment and always runs AB.
Values, blocks, and seed fields look plausible, but every realized order is AB.
Under increasing drift, B is always measured later and inherits drift; planted
−1 ms B effect reverses to a positive B−A estimate.

Black-box evidence: `simulate(..., randomized=True)` lacks both AB and BA orders,
changes no assignment with seed, and produces biased estimate under monotonic
drift. `test_randomization_resists_drift_but_fixed_order_reverses` detects effect;
assignment test identifies mechanism.
