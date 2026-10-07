# LAB C BREAK Key

No separate planted bug in LAB C; module-level BREAK artifact ships in LAB A
(and an extra seeded-assignment bug ships in LAB B). LAB C's deliberate break
is the named `naive_request_interval`: treating correlated requests as
independent units. It is retained for comparison, not presented as valid
process-level inference.

Expected observation: duplicating requests within each unchanged process
shrinks naive request-level interval but leaves process-level t interval and
whole-process bootstrap unchanged. Increasing independent process count at a
similar total request count narrows process-level uncertainty.
