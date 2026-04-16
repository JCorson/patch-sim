"""Tests for the patch_sim package's public API (__all__)."""

import patch_sim


def test_all_exports_are_accessible():
    """Every name listed in __all__ must be accessible via getattr(patch_sim, name)."""
    for name in patch_sim.__all__:
        assert hasattr(patch_sim, name), (
            f"{name!r} is in __all__ but not accessible on patch_sim"
        )
        obj = getattr(patch_sim, name)
        assert obj is not None, f"{name!r} resolved to None"
