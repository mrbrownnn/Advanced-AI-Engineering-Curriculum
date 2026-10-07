"""LAB A starter: sector-transaction model, useful bandwidth, CPU tiled matmul oracle.

Declared model (D, not a hardware specification):
- one warp instruction requests `access_bytes` per active lane at byte addresses;
- the memory system moves whole aligned sectors of `sector_bytes` (a parameter,
  default 32; fixed transaction rules are architecture-specific);
- sectors are counted per warp instruction with no reuse between instructions,
  so `moved_bytes` is request-level traffic (L1->L2 style), an upper bound for
  DRAM traffic only when nothing is reused in a cache.

Fill every TODO. Keep the signatures; tests import these names.
"""
import numpy as np

DEFAULT_SECTOR_BYTES = 32
WARP_SIZE = 32
UNITS = {"B/s": 1, "MB/s": 10**6, "GB/s": 10**9, "MiB/s": 2**20, "GiB/s": 2**30}


def strided_addresses(n_lanes, elem_bytes, stride_elems=1, offset_elems=0, base=0):
    """Byte address of lane i = base + (offset_elems + i * stride_elems) * elem_bytes."""
    if n_lanes <= 0 or elem_bytes <= 0 or stride_elems <= 0 or offset_elems < 0 or base < 0:
        raise ValueError("n_lanes, elem_bytes, stride_elems must be > 0; offset, base >= 0")
    return [base + (offset_elems + i * stride_elems) * elem_bytes for i in range(n_lanes)]


def aos_addresses(n_lanes, struct_bytes, field_offset, field_bytes, base=0):
    """Lane i reads one field of struct i in an array-of-structs."""
    if field_offset < 0 or field_bytes <= 0 or field_offset + field_bytes > struct_bytes:
        raise ValueError("field must lie inside the struct")
    return [base + i * struct_bytes + field_offset for i in range(n_lanes)]


def soa_addresses(n_lanes, field_bytes, base=0):
    """Lane i reads element i of one field array in a struct-of-arrays."""
    return strided_addresses(n_lanes, field_bytes, 1, 0, base)


def sectors_touched(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES):
    """Return the set of aligned sector indices covered by every byte range [a, a + access_bytes).

    An access whose bytes cross a sector boundary touches more than one sector.
    Raise ValueError if access_bytes or sector_bytes is not positive.
    """
    raise NotImplementedError("TODO: sectors_touched")


def warp_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES):
    """One warp instruction. Return dict with keys:

    useful_bytes  distinct requested bytes (lanes reading the same byte count once)
    moved_bytes   whole sectors moved * sector_bytes
    sectors       number of distinct sectors
    efficiency    useful_bytes / moved_bytes
    """
    raise NotImplementedError("TODO: warp_traffic")


def pattern_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES, warp_size=WARP_SIZE):
    """Split lanes into consecutive warps of `warp_size`, sum warp_traffic fields, recompute efficiency."""
    raise NotImplementedError("TODO: pattern_traffic")


def useful_bandwidth(useful_bytes, seconds, unit="GB/s"):
    """(B_r + B_w) / t in `unit`. Bytes and seconds in, divide by the unit's divisor exactly once.

    Raise ValueError for seconds <= 0 or an unknown unit.
    """
    raise NotImplementedError("TODO: useful_bandwidth")


def vector_add(a, b):
    """Elementwise a + b with a shape assertion. Return a NumPy array."""
    raise NotImplementedError("TODO: vector_add")


def tiled_matmul(a, b, tile=16):
    """C = A @ B computed tile by tile (loops over i, j, k tiles), accumulating in at least float32.

    Assert 2-D inputs and matching inner dimension (AssertionError on mismatch).
    Handle shapes that are not multiples of `tile`.
    """
    raise NotImplementedError("TODO: tiled_matmul")
