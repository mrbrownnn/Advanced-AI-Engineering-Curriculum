"""Deterministic stand-in for an asynchronous GPU: host clock + per-stream FIFO queues.

Provided, not a TODO: this is the "hardware" for CPU tests. Semantics mirror the
CUDA-like rules in Lesson 2.3, simplified (D, not a device model):

- launch() costs the host `launch_cost` seconds and returns BEFORE the kernel runs;
  the first launch also pays `first_launch_cost` (cold start: context/init).
- an op reaches the device `queue_latency` after it was enqueued; it starts when it
  has arrived AND its stream is free. Streams run concurrently (no resource limit).
- record_event(stream) completes when all earlier work on THAT stream completes.
- synchronize() blocks the host until every stream drains, plus `sync_return`.
- Event.elapsed(end) raises RuntimeError("event not ready") if the host has not
  yet observed completion of `end` -- reading device time requires a boundary.
"""
from collections import defaultdict


class SimClock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def now(self):
        return self.t

    def advance(self, dt):
        if dt < 0:
            raise ValueError("time does not run backwards")
        self.t += dt


class Event:
    def __init__(self, dev, ts):
        self._dev = dev
        self.ts = ts

    def done(self):
        return self._dev.clock.now() >= self.ts

    def elapsed(self, end):
        if not (self.done() and end.done()):
            raise RuntimeError("event not ready: synchronize on the end event before reading it")
        return end.ts - self.ts


class SimDevice:
    def __init__(self, clock, launch_cost=5e-6, queue_latency=5e-6, sync_return=5e-6, first_launch_cost=0.0):
        self.clock = clock
        self.launch_cost = launch_cost
        self.queue_latency = queue_latency
        self.sync_return = sync_return
        self.first_launch_cost = first_launch_cost
        self.stream_free = defaultdict(float)
        self.kernels = []  # (stream, name, start, end) -- the "timeline"

    def launch(self, duration, stream=0, name="kernel"):
        cost = self.launch_cost + (self.first_launch_cost if not self.kernels else 0.0)
        self.clock.advance(cost)
        start = max(self.clock.now() + self.queue_latency, self.stream_free[stream])
        end = start + duration
        self.stream_free[stream] = end
        self.kernels.append((stream, name, start, end))
        return end

    def record_event(self, stream=0):
        ts = max(self.clock.now() + self.queue_latency, self.stream_free[stream])
        self.stream_free[stream] = ts
        return Event(self, ts)

    def _block_until(self, t):
        self.clock.advance(max(0.0, t - self.clock.now()) + self.sync_return)

    def synchronize(self):
        self._block_until(max(self.stream_free.values(), default=0.0))

    def stream_synchronize(self, stream=0):
        self._block_until(self.stream_free[stream])

    def event_synchronize(self, event):
        self._block_until(event.ts)


def kernels_work(n, duration, stream=0):
    """Workload factory: n identical kernels on one stream."""
    def work(dev):
        for _ in range(n):
            dev.launch(duration, stream)
    return work
