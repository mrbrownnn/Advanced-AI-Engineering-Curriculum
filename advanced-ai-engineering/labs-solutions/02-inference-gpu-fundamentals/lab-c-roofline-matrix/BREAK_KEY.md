# LAB C — Instructor Key

No planted `bugged_*.py` in this lab; module BREAK requirement is satisfied by Labs A and B.

## Checks the reference must satisfy

- FMA = 2 FLOPs: GEMM W = `2*M*N*K`.
- One-pass Q = `s*(M*K + K*N + M*N)` at the named boundary.
- `bound` refuses a compute ceiling for another instruction path and a bandwidth ceiling for another hierarchy boundary.
- Above `I* = P_peak / beta`, compute time dominates; below it, memory time dominates.
- More bandwidth cannot turn a compute-screened workload into a memory-screened one.
- `qualify` never relabels the basic screen: a point far below both ceilings raises an unmodeled-limiter flag.
- Tiny high-intensity work can still underfill the GPU; intensity does not encode grid size.

`t4_nameplate` values, measured ceilings, instruction paths, and all Colab outcomes are `TODO_VERIFY` until a pilot run. The Lesson 2.4 exercise ceiling set is not a GPU claim.
