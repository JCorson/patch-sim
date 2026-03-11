"""
Tests for the ap_sim package's public API (__all__).
"""

import ap_sim


def test_all_exports_are_accessible():
    """Every name listed in __all__ must be accessible via getattr(ap_sim, name)."""
    for name in ap_sim.__all__:
        assert hasattr(ap_sim, name), (
            f"{name!r} is in __all__ but not accessible on ap_sim"
        )
        obj = getattr(ap_sim, name)
        assert obj is not None, f"{name!r} resolved to None"
