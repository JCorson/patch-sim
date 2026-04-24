"""Smoke tests for UI component instantiation."""

import pytest

pytest.importorskip("reflex")


def _true_branch_children(cond_component):
    """Return the children of the true branch inside an ``rx.cond`` component.

    Args:
        cond_component: The component returned by ``rx.cond``, which is a
            ``Fragment`` wrapping a single ``Cond`` node.

    Returns:
        The children list of the true-branch ``Fragment``.
    """
    cond_node = cond_component.children[0]
    true_fragment = cond_node.children[0]
    return true_fragment.children


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


def test_channel_trace_group_excludes_current_when_flag_false():
    """_channel_trace_group with include_current=False must omit the current checkbox.

    The true branch of the ``rx.cond`` should have 3 children (separator,
    section header, gating checkbox) rather than 4.
    """
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[0]
    comp = _channel_trace_group(*spec, include_current=False)
    children = _true_branch_children(comp)
    assert len(children) == 3, (
        f"Expected 3 children (separator, header, gating); got {len(children)}"
    )


def test_channel_trace_group_includes_current_by_default():
    """_channel_trace_group must include the current checkbox by default.

    The true branch of the ``rx.cond`` should have 4 children (separator,
    section header, current checkbox, gating checkbox).
    """
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[0]
    comp = _channel_trace_group(*spec)
    children = _true_branch_children(comp)
    assert len(children) == 4, (
        f"Expected 4 children (separator, header, current, gating); got {len(children)}"
    )


def test_cc_additional_channels_section_gating_only():
    """_additional_channels_section(include_current=False) must render without error.

    Every channel group in the CC section must have exactly 3 children in its
    true branch — the current checkbox must be absent.
    """
    from patch_sim_ui.channels import ADDITIONAL_CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for i, ch in enumerate(ADDITIONAL_CHANNELS):
        spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[i]
        comp = _channel_trace_group(*spec, include_current=False)
        children = _true_branch_children(comp)
        assert len(children) == 3, (
            f"Channel {ch.label!r}: expected 3 children in CC mode, got {len(children)}"
        )


def test_vc_additional_channels_section_includes_current():
    """_additional_channels_section() (VC default) must include the current checkbox.

    Every channel group in the VC section must have exactly 4 children in its
    true branch — both the current and gating checkboxes must be present.
    """
    from patch_sim_ui.channels import ADDITIONAL_CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for i, ch in enumerate(ADDITIONAL_CHANNELS):
        spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[i]
        comp = _channel_trace_group(*spec, include_current=True)
        children = _true_branch_children(comp)
        assert len(children) == 4, (
            f"Channel {ch.label!r}: expected 4 children in VC mode, got {len(children)}"
        )


def test_log_panel_renders_without_error():
    """Instantiating log_panel must not raise a TypeError."""
    from patch_sim_ui.components.log_panel import log_panel

    log_panel()


def test_trace_display_renders_without_error():
    """Instantiating trace_display must not raise a TypeError."""
    from patch_sim_ui.components.trace_display import trace_display

    trace_display()
