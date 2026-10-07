# LAB D — Instructor Key

No planted `bugged_*.py` in this lab; module BREAK requirement is satisfied by Labs A and B.

## Synthetic truth table

- host launch gap: time moves between kernels; kernel durations/counters stay stable.
- synchronization: gap time rises **and sync calls appear**; this discriminates it from host launch gaps.
- small grid: mean grid blocks fall; intensity alone cannot expose this.
- strided loads: kernel time and sectors/request rise; spills/registers stay stable.
- register pressure: kernel time, registers/thread, and spills rise; sectors/request stays stable.
- divergence: branch efficiency falls.
- background activity: another pid contributes kernel time.
- clock throttle: SM clock falls and kernel duration rises.

Without kernel counters, strided loads and register pressure remain unresolved. Correct behavior is abstention plus a request for the discriminating measurement or a controlled intervention — never guessing from utilization. The synthetic data contain one cause at a time; interacting-cause diagnosis and all Colab outcomes remain `TODO_VERIFY`.
