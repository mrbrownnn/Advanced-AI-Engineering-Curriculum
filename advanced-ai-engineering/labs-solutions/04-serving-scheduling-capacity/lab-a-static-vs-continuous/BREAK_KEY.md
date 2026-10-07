# BREAK key — LAB A

`starter/bugged_metrics.py` measures TTFT from `admit` rather than request `arrival`. The omitted queue interval is a coordinated-omission cousin: overload can lengthen the queue while the reported value stays low. `test_break.py` supplies nonzero queue residence and requires the declared client-arrival boundary. Fix only that subtraction, then re-run the same workload and explain which SLO population changes.
