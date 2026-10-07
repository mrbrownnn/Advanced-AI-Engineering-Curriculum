"""Colab-only adapter: exposes torch.cuda through the same interface as starter/simdevice.py,
so timing.py runs unchanged on a real GPU. Not imported by tests. Requires CUDA torch.

    dev = TorchDevice()
    work = lambda d: [d.launch(lambda: a @ b) for _ in range(10)]
"""
import torch


class _Event:
    def __init__(self, ev):
        self.ev = ev

    def elapsed(self, end):
        return self.ev.elapsed_time(end.ev) / 1e3  # CUDA events report milliseconds


class TorchDevice:
    def __init__(self, n_streams=2):
        assert torch.cuda.is_available(), "TorchDevice needs a CUDA GPU (run on Colab, not locally)"
        self.streams = [torch.cuda.current_stream()] + [torch.cuda.Stream() for _ in range(n_streams - 1)]

    def launch(self, fn, stream=0, name="op"):
        with torch.cuda.stream(self.streams[stream]):
            return fn()

    def record_event(self, stream=0):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self.streams[stream])
        return _Event(ev)

    def synchronize(self):
        torch.cuda.synchronize()

    def stream_synchronize(self, stream=0):
        self.streams[stream].synchronize()

    def event_synchronize(self, event):
        event.ev.synchronize()
