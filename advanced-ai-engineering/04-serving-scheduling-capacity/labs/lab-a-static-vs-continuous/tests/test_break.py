import bugged_metrics as bugged


def test_bugged_ttft_includes_queue_residence():
    r = bugged.Record(bugged.Request(7, 100.0, 16, 2), admit=220.0, tokens=[285.0, 310.0])
    assert bugged.ttft(r) == 185.0, "HINT: Use one declared arrival boundary; admission is t1, not t0."
