# BREAK key — LAB D

A recompute victim loses resident KV. On resumption it must re-prefill `N_in + emitted_prefix`; those tokens contribute to both elapsed step cost and `wasted_tokens`. Omitting that cost makes preemption look free. Thrashing requires aligned evidence: allocation failures, preemption churn, and useful-token progress below a declared baseline. Churn alone is not sufficient.
