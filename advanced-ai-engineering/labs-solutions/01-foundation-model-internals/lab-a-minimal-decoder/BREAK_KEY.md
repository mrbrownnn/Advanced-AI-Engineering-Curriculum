# LAB A BREAK key

`starter/bugged_mask.py` masks `k_pos >= q_pos`, excluding the diagonal. Correct causal semantics permit `j <= i`; only `j > i` is `-inf`. Shape tests pass, but `tests/test_components.py::test_causal_mask_semantics` fails at the diagonal check. In an otherwise complete decoder, each position cannot attend to its own K/V and position 0 creates an all-`-inf` attention row, often producing NaNs.

Diagnostic: inspect the additive mask row by row before final logits. Immediate fix: change `>=` to `>`. Prevention: keep diagonal, orientation, cached-prefix, and future-perturbation invariants.
