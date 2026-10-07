"""BREAK exercise: a synchronized timer that looks right and has ONE real bug.

Do not diff it. Use simdevice to build a scenario where its answer is wrong,
explain the mechanism, and record it in report.md section 6.
"""


def time_synced(dev, timer, work):
    t0 = timer()
    work(dev)
    dev.synchronize()
    return timer() - t0
