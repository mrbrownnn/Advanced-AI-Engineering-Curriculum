# BREAK key — LAB B

The explicit break is `C=32`. Under the supplied model, each chunk creates another scheduled step and pays `per_chunk`; it may lower an individual injected decode stall while increasing prompt TTFT and reducing throughput. Compare step count and separated cost components, not ITL alone. Counterexample: short prompts or zero per-chunk overhead can make the proposed degradation disappear.
