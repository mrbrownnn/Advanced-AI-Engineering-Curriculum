"""BREAK exercise: a complete, working-looking transaction counter with ONE real bug.

Do not diff it against your own code. Instrument it: feed it address patterns
whose correct sector count you can derive by hand, and find where it disagrees.
Record what you found in report.md section 6.
"""
from transactions import (DEFAULT_SECTOR_BYTES, WARP_SIZE, aos_addresses,  # noqa: F401
                          soa_addresses, strided_addresses)


def sectors_touched(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES):
    if access_bytes <= 0 or sector_bytes <= 0:
        raise ValueError("access_bytes and sector_bytes must be > 0")
    return {a // sector_bytes for a in addresses}


def warp_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES):
    useful = len({a + k for a in addresses for k in range(access_bytes)})
    sectors = len(sectors_touched(addresses, access_bytes, sector_bytes))
    moved = sectors * sector_bytes
    return {"useful_bytes": useful, "moved_bytes": moved, "sectors": sectors,
            "efficiency": useful / moved if moved else 1.0}


def pattern_traffic(addresses, access_bytes, sector_bytes=DEFAULT_SECTOR_BYTES, warp_size=WARP_SIZE):
    useful = moved = sectors = 0
    for i in range(0, len(addresses), warp_size):
        t = warp_traffic(addresses[i:i + warp_size], access_bytes, sector_bytes)
        useful += t["useful_bytes"]
        moved += t["moved_bytes"]
        sectors += t["sectors"]
    return {"useful_bytes": useful, "moved_bytes": moved, "sectors": sectors,
            "efficiency": useful / moved if moved else 1.0}
