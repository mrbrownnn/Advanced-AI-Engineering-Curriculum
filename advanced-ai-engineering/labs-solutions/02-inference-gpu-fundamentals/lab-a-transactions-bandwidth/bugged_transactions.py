"""Fixed version of the BREAK module: a range access touches every sector it spans.

See BREAK_KEY.md for the planted bug.
"""
from transactions import (DEFAULT_SECTOR_BYTES, WARP_SIZE, aos_addresses,  # noqa: F401
                          pattern_traffic, sectors_touched, soa_addresses,
                          strided_addresses, warp_traffic)
