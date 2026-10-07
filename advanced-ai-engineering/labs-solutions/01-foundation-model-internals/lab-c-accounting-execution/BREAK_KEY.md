# LAB C BREAK key

No planted starter bug belongs to this lab; Module 01 BREAK requirement is met by Labs A and B. Required Lab C fault injections are configuration changes, not hidden code: turn on `bias`, toggle `tie_embeddings`, set `adapter_rank`, and enable `experts`/`experts_per_token`. Tests demand each corresponding manifest delta reconcile with instantiated tensors.

Common intentional failure: count `sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))` on a tied model. This double-counts aliased embedding/head storage. `actual_parameter_count(..., unique_storage=True)` must deduplicate using storage identity.
