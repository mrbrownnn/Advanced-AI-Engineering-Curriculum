# LAB A BREAK Key

`starter/bugged_harness.py` contains one planted bug: `summarize()` always uses `event.actual` as latency origin. It accepts `origin="intended"` but silently ignores that contract. Same delayed event therefore has same reported latency under both origins, hiding issue lag.

Expected black-box evidence: keep event end and SUT work fixed, score an event with `intended < actual`, then compare origin policies. Intended-origin latency must be larger by `actual - intended`. `test_origin_changes_the_number` detects violation.

Expected stall observation: response coupling records far fewer intended arrivals in stall window and much lower actual-origin P95 than independent open-loop schedule. This is coordinated omission, not faster SUT work.
