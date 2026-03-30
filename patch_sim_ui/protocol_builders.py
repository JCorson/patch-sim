"""Re-exports of protocol builder functions from the core library.

The implementations live in ``patch_sim.protocols.builders``.  This module
exists for backward compatibility so that existing UI and test imports
continue to work without changes.
"""

from patch_sim.protocols.builders import (  # noqa: F401
    build_current_protocol,
    build_voltage_protocol,
)

__all__ = ["build_current_protocol", "build_voltage_protocol"]
