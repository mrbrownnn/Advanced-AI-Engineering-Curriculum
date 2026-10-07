# LAB A — Instructor / BREAK Key

Open only after submitting the lab report.

## Break result

`fixtures.make_trace.equal_fixed(n=8, length=256)`, capacity `8*256`,
`block_size=1`, reservation `max_len == final length`: zero reservation waste,
zero internal fragmentation, zero capacity/contiguous failures, at most one free
hole. Nonzero waste indicates category arithmetic or an allocator-hole invariant is
wrong.

## Distinguishing failures

With extents `[0,4)`, `[4,6)`, `[6,10)`, releasing the first and last leaves 8
slots free but no 6-slot hole: request size 6 fails `contiguous`; size 9 fails
`capacity`. Paging addresses the contiguous logical-placement requirement, not
framework/CUDA allocator fragmentation.
