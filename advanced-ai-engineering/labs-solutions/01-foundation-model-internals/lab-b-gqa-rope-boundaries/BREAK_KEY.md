# LAB B BREAK key

`starter/bugged_grouping.py` uses `x.repeat(1, group_size, 1, 1)`. For two KV heads and group size two this maps output heads `[0,1,2,3]` to KV heads `[0,1,0,1]`. Expected contiguous mapping is `[0,0,1,1]`, or group `h // group_size`.

Shapes and counts pass. Per-head parity against `torch.repeat_interleave(x, group_size, dim=1)` fails first at output head 1. Fix with `repeat_interleave` or an `expand` then reshape preserving `[Hkv, group]` order. Prevention: test head identity, not only shape.
