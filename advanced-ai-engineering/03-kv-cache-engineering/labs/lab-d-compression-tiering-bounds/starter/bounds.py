"""Compression, tiering and disaggregation screening bounds (Lessons 3.8-3.10). Fill the TODOs.

Every function returns a LOWER bound or an optimistic UPPER bound; none is a measurement.
Units: bytes, seconds. Bandwidth carries its unit explicitly to stop GiB/GB mixing.
"""
from __future__ import annotations

from dataclasses import dataclass

KIB, MIB, GIB = 1024, 1024 ** 2, 1024 ** 3
GB = 1000 ** 3
UNITS = {"GiB/s": GIB, "GB/s": GB, "MiB/s": MIB, "MB/s": 1000 ** 2}


def payload_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: float) -> float:
    """Nominal K+V payload at `bits` per element (no metadata)."""
    raise NotImplementedError("HINT: TODO")


def scale_bytes_per_token(layers: int, kv_heads: int, scale_bytes: int, group_tokens: int) -> float:
    """Quantization metadata: one scale per (layer, K or V, KV head, group of group_tokens tokens).
    Raise ValueError if group_tokens < 1."""
    raise NotImplementedError("HINT: TODO")


def concurrency_bound(kv_budget_bytes: float, tokens_per_request: int, bytes_per_token: float) -> int:
    """Payload-only optimistic upper bound (integer)."""
    raise NotImplementedError("HINT: TODO")


@dataclass(frozen=True)
class Bandwidth:
    value: float
    unit: str  # one of UNITS

    def bytes_per_s(self) -> float:
        """Convert to bytes/s; raise ValueError for an unknown unit."""
        raise NotImplementedError("HINT: TODO")


def transfer_lower_bound_s(payload_bytes: float, bw: Bandwidth) -> float:
    """Serialization lower bound. Refuse (TypeError) a bare number: units must be explicit."""
    raise NotImplementedError("HINT: TODO")


def critical_path_s(payload_bytes: float, bw: Bandwidth, setup_s: float = 0.0,
                    overlap_fraction: float = 0.0) -> float:
    """Setup + the non-overlapped share of serialization. overlap_fraction in [0, 1] else ValueError."""
    raise NotImplementedError("HINT: TODO")


def setup_dominated(payload_bytes: float, bw: Bandwidth, setup_s: float) -> bool:
    """True when fixed setup exceeds the serialization time."""
    raise NotImplementedError("HINT: TODO")


def disagg_ttft_s(prefill_s: float, payload_bytes: float, bw: Bandwidth, setup_s: float,
                  overlap_fraction: float, first_decode_s: float) -> float:
    """Simplified prefill->decode handoff: prefill, non-overlapped transfer, first decode step."""
    raise NotImplementedError("HINT: TODO")


@dataclass
class GateResult:
    passed: bool
    metric: float
    threshold: float
    reason: str


def quality_gate(baseline_scores, candidate_scores, max_drop: float) -> GateResult:
    """Paired gate: mean per-item (baseline - candidate) must not exceed max_drop.
    Raise ValueError for unequal or empty inputs."""
    raise NotImplementedError("HINT: TODO")


def capacity_claim(bound_requests: int, gate: GateResult) -> int | None:
    """A capacity claim survives only if the quality gate passed; otherwise return None."""
    raise NotImplementedError("HINT: TODO")


def retained_fraction_scores(scores_by_position, keep_fraction: float):
    """Toy lossy-retention stub (provided): an item's score survives only if its evidence
    position is within the retained most-recent window. Positions are 0..1 (0 = oldest)."""
    return [s if pos >= 1.0 - keep_fraction else 0.0 for s, pos in scores_by_position]
