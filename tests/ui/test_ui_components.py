"""Smoke tests for UI component instantiation."""

import pytest

pytest.importorskip("reflex")


def _label_present(component, label: str) -> bool:
    """Return True if ``label`` appears as visible text in the component tree.

    Checks the component's repr, which Reflex populates with all rendered text
    contents including checkbox labels.

    Args:
        component: Any Reflex component.
        label: The label string to search for.

    Returns:
        True when the label is found in the rendered component output.
    """
    return label in repr(component)


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


def test_analysis_sidebar_renders_with_burst_data():
    """analysis_sidebar must render the new burst panel without error.

    The sidebar wires the burst summary into the Analysis sub-tab and the
    per-burst table into the new Bursts sub-tab; this smoke test ensures
    the surrounding tab structure still builds.
    """
    from patch_sim_ui.components.metrics_panel import analysis_sidebar

    analysis_sidebar()


def test_channel_trace_group_excludes_current_when_flag_false():
    """_channel_trace_group with include_current=False must omit the current label.

    The current checkbox label (e.g. ``"I_Ih"``) must not appear in the rendered
    component while the gating label must still be present.
    """
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[0]
    current_label, gating_label = spec[2], spec[5]
    comp = _channel_trace_group(*spec, include_current=False)
    assert not _label_present(comp, current_label), (
        f"Current label {current_label!r} must not appear when include_current=False"
    )
    assert _label_present(comp, gating_label), (
        f"Gating label {gating_label!r} must still appear when include_current=False"
    )


def test_channel_trace_group_includes_current_by_default():
    """_channel_trace_group must include the current checkbox by default.

    Both the current label (e.g. ``"I_Ih"``) and the gating label must appear
    in the rendered component.
    """
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[0]
    current_label, gating_label = spec[2], spec[5]
    comp = _channel_trace_group(*spec)
    assert _label_present(comp, current_label), (
        f"Current label {current_label!r} must appear by default"
    )
    assert _label_present(comp, gating_label), (
        f"Gating label {gating_label!r} must appear by default"
    )


def test_cc_additional_channels_section_gating_only():
    """_additional_channels_section(include_current=False) must hide all current labels.

    For every additional channel the current label (e.g. ``"I_Ih"``) must not
    appear in the rendered group while the gating label must remain visible.
    """
    from patch_sim_ui.channels import ADDITIONAL_CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for i, ch in enumerate(ADDITIONAL_CHANNELS):
        spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[i]
        comp = _channel_trace_group(*spec, include_current=False)
        assert not _label_present(comp, ch.current_label), (
            f"Channel {ch.label!r}: {ch.current_label!r} must be absent in CC mode"
        )
        assert _label_present(comp, ch.gating_label), (
            f"Channel {ch.label!r}: {ch.gating_label!r} must be present in CC mode"
        )


def test_vc_additional_channels_section_includes_current():
    """_additional_channels_section() (VC default) must show both labels.

    For every additional channel both the current label (e.g. ``"I_Ih"``) and
    the gating label must appear in the rendered group.
    """
    from patch_sim_ui.channels import ADDITIONAL_CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _ADDITIONAL_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for i, ch in enumerate(ADDITIONAL_CHANNELS):
        spec = _ADDITIONAL_CHANNEL_TRACE_SPECS[i]
        comp = _channel_trace_group(*spec, include_current=True)
        assert _label_present(comp, ch.current_label), (
            f"Channel {ch.label!r}: {ch.current_label!r} must be present in VC mode"
        )
        assert _label_present(comp, ch.gating_label), (
            f"Channel {ch.label!r}: {ch.gating_label!r} must be present in VC mode"
        )


def test_log_panel_renders_without_error():
    """Instantiating log_panel must not raise a TypeError."""
    from patch_sim_ui.components.log_panel import log_panel

    log_panel()


def test_trace_display_renders_without_error():
    """Instantiating trace_display must not raise a TypeError."""
    from patch_sim_ui.components.trace_display import trace_display

    trace_display()
