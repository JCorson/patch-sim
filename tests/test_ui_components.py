"""Smoke tests for UI component instantiation."""

import pytest

pytest.importorskip("reflex")


def test_neuron_panel_renders_without_error():
    """Instantiating neuron_panel must not raise a TypeError."""
    from patch_sim_ui.components.neuron_panel import neuron_panel

    neuron_panel()  # must not raise TypeError


def test_protocol_panel_renders_without_error():
    """Instantiating protocol_panel must not raise a TypeError."""
    from patch_sim_ui.components.protocol_panel import protocol_panel

    protocol_panel()


def test_index_page_renders_without_error():
    """Instantiating the main index page must not raise."""
    from patch_sim_ui.patch_sim_ui import index

    index()
