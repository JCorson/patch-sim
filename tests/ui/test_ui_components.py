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


def test_sweep_manager_renders_without_error():
    """Instantiating sweep_manager must not raise a TypeError."""
    from patch_sim_ui.components.sweep_manager import sweep_manager

    sweep_manager()


def test_log_panel_renders_without_error():
    """Instantiating log_panel must not raise a TypeError."""
    from patch_sim_ui.components.log_panel import log_panel

    log_panel()


def test_trace_display_renders_without_error():
    """Instantiating trace_display must not raise a TypeError."""
    from patch_sim_ui.components.trace_display import trace_display

    trace_display()
