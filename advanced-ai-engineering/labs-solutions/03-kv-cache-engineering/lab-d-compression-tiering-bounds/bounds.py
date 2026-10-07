"""Compression, tiering and disaggregation screening bounds (Lessons 3.8-3.10). Reference.

Every function returns a LOWER bound or an optimistic UPPER bound; none is a measurement.
Units: bytes, seconds. Bandwidth carries its unit explicitly to stop GiB/GB mixing.
"""
from __future__ import annotations

from dataclasses import dataclass

KIB, MIB, GIB = 1024, 1024 ** 2, 1024 ** 3
GB = 1000 ** 3
UNITS = {"GiB/s": GIB, "GB/s": GB, "MiB/s": MIB, "MB/s": 1000 ** 2}


def payload_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: float) -> float:
    return 2 * layers * kv_heads * head_dim * bits / 8


def scale_bytes_per_token(layers: int, kv_heads: int, scale_bytes: int, group_tokens: int) -> float:
    """One scale per (layer, K or V, KV head, group of group_tokens tokens)."""
    if group_tokens < 1:
        raise ValueError("group_tokens must be >= 1")
    return layers * 2 * kv_heads * scale_bytes / group_tokens


def concurrency_bound(kv_budget_bytes: float, tokens_per_request: int, bytes_per_token: float) -> int:
    return int(kv_budget_bytes // (tokens_per_request * bytes_per_token))


@dataclass(frozen=True)
class Bandwidth:
    value: float
    unit: str

    def bytes_per_s(self) -> float:
        if self.unit not in UNITS:
            raise ValueError(f"unknown bandwidth unit {self.unit!r}; use one of {sorted(UNITS)}")
        return self.value * UNITS[self.unit]


def transfer_lower_bound_s(payload_bytes: float, bw: Bandwidth) -> float:
    if not isinstance(bw, Bandwidth):
        raise TypeError("pass Bandwidth(value, unit), not a bare number")
    return payload_bytes / bw.bytes_per_s()


def critical_path_s(payload_bytes: float, bw: Bandwidth, setup_s: float = 0.0,
                    overlap_fraction: float = 0.0) -> float:
    """Setup + the non-overlapped share of serialization."""
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    return setup_s + (1.0 - overlap_fraction) * transfer_lower_bound_s(payload_bytes, bw)


def setup_dominated(payload_bytes: float, bw: Bandwidth, setup_s: float) -> bool:
    return setup_s > transfer_lower_bound_s(payload_bytes, bw)


def disagg_ttft_s(prefill_s: float, payload_bytes: float, bw: Bandwidth, setup_s: float,
                  overlap_fraction: float, first_decode_s: float) -> float:
    """Simplified prefill->decode handoff: prefill, non-overlapped transfer, first decode step."""
    return prefill_s + critical_path_s(payload_bytes, bw, setup_s, overlap_fraction) + first_decode_s


@dataclass
class GateResult:
    passed: bool
    metric: float
    threshold: float
    reason: str


def quality_gate(baseline_scores, candidate_scores, max_drop: float) -> GateResult:
    """Paired gate: mean per-item drop vs baseline must not exceed max_drop."""
    if len(baseline_scores) != len(candidate_scores) or not baseline_scores:
        raise ValueError("need equal-length, non-empty paired scores")
    drop = sum(b - c for b, c in zip(baseline_scores, candidate_scores)) / len(baseline_scores)
    ok = drop <= max_drop
    return GateResult(ok, drop, max_drop, "ok" if ok else f"mean drop {drop:.4f} > {max_drop}")


def capacity_claim(bound_requests: int, gate: GateResult) -> int | None:
    """A capacity claim survives only if the quality gate passed; otherwise no claim (None)."""
    return bound_requests if gate.passed else None


def retained_fraction_scores(scores_by_position, keep_fraction: float):
    """Toy lossy-retention stub: an item's score survives only if its evidence position
    is within the retained most-recent window. Positions are 0..1 (0 = oldest)."""
    return [s if pos >= 1.0 - keep_fraction else 0.0 for s, pos in scores_by_position]
