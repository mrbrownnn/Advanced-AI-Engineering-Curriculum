"""LAB A reference: sector-transaction model, useful bandwidth, CPU tiled matmul oracle.

Declared model (D, not a hardware specification):
- one warp instruction requests `access_bytes` per active lane at byte addresses;
- the memory system moves whole aligned sectors of `sector_bytes` (a parameter,
  default 32; fixed transaction rules are architecture-specific);
- sectors are counted per warp instruction with no reuse between instructions,
  so `moved_bytes` is request-level traffic (L1->L2 style), an upper bound for
  DRAM traffic only when nothing is reused in a cache.
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
    """Set of aligned sector indices covered by every [a, a + access_bytes)."""
    if access_bytes <= 0 or sector_bytes <= 0:
        raise ValueError("access_bytes and sector_bytes must be > 0")
    sectors = set()
    for a in addresses:
        sectors.update(range(a // sector_bytes, (a + access_bytes - 1) // sector_bytes + 1))
    return sectors


def warp_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES):
    """Useful (distinct requested) bytes vs moved (whole-sector) bytes for one warp instruction."""
    useful = len({a + k for a in addresses for k in range(access_bytes)})
    sectors = len(sectors_touched(addresses, access_bytes, sector_bytes))
    moved = sectors * sector_bytes
    return {"useful_bytes": useful, "moved_bytes": moved, "sectors": sectors,
            "efficiency": useful / moved if moved else 1.0}


def pattern_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES, warp_size=WARP_SIZE):
    """Sum warp_traffic over consecutive groups of `warp_size` lanes (one instruction per warp)."""
    if warp_size <= 0:
        raise ValueError("warp_size must be > 0")
    useful = moved = sectors = 0
    for i in range(0, len(addresses), warp_size):
        t = warp_traffic(addresses[i:i + warp_size], access_bytes, sector_bytes)
        useful += t["useful_bytes"]
        moved += t["moved_bytes"]
        sectors += t["sectors"]
    return {"useful_bytes": useful, "moved_bytes": moved, "sectors": sectors,
            "efficiency": useful / moved if moved else 1.0}


def useful_bandwidth(useful_bytes, seconds, unit="GB/s"):
    """(B_r + B_w) / t, converted to `unit` by dividing by its exact byte divisor once."""
    if seconds <= 0:
        raise ValueError("seconds must be > 0 (synchronized elapsed time)")
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; use one of {sorted(UNITS)}")
    return useful_bytes / seconds / UNITS[unit]


def vector_add(a, b):
    """Elementwise add with shape assertion; used as the simple arithmetic-preserving kernel oracle."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    return a + b


def tiled_matmul(a, b, tile=16):
    """C = A @ B computed tile by tile, with shape and bounds assertions (correctness oracle)."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.ndim == 2 and b.ndim == 2, "A and B must be 2-D"
    assert a.shape[1] == b.shape[0], f"inner dims differ: {a.shape} @ {b.shape}"
    assert tile > 0, "tile must be > 0"
    m, k = a.shape
    n = b.shape[1]
    acc = np.result_type(a, b, np.float32)
    c = np.zeros((m, n), dtype=acc)
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            for k0 in range(0, k, tile):
                i1, j1, k1 = min(i0 + tile, m), min(j0 + tile, n), min(k0 + tile, k)
                c[i0:i1, j0:j1] += a[i0:i1, k0:k1].astype(acc) @ b[k0:k1, j0:j1].astype(acc)
    return c
