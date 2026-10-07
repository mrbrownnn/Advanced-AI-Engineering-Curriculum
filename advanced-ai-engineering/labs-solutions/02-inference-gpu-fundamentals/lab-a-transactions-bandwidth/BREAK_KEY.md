# LAB A — BREAK Key

## Planted bug

`starter/bugged_transactions.py:sectors_touched` counts only `a // sector_bytes`, the sector containing each access's **first byte**. An access is a range `[a, a + access_bytes)` and can cross into another sector. Thus a byte-misaligned warp starting at byte 2 and reading 32 contiguous 4-byte values spans five 32-byte sectors, but the bug reports four.

## Discriminating probe

Start with one 4-byte access at byte 30. Correct touched sectors: `{0, 1}`. Sweep start byte 0–31 and observe the count change at start 29. This isolates address-range accounting from stride or cache effects.

## Fix

For every access add all sector indices from `a // sector_bytes` through `(a + access_bytes - 1) // sector_bytes`, inclusive.

## Expected observations (model only)

Aligned stride-1 has efficiency 1.0 in the declared sector model. Increasing stride cannot increase efficiency and saturates at one sector per lane. AoS field-read traffic matches a strided scalar read. These are model outputs; Colab measurements remain `TODO_VERIFY`.
