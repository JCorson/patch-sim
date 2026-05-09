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


def test_protocol_param_schema_covers_all_protocols():
    """Schema keys must match every (clamp_mode, protocol_type) pair.

    Every protocol exposed via ``CURRENT_PROTOCOLS`` / ``VOLTAGE_PROTOCOLS``
    must have a schema entry, and the schema must not contain entries for
    protocols that are not advertised in the UI dropdown.  This catches the
    "added a protocol but forgot the schema entry" failure mode.
    """
    from patch_sim.constants import (
        CURRENT_CLAMP,
        CURRENT_PROTOCOLS,
        VOLTAGE_CLAMP,
        VOLTAGE_PROTOCOLS,
    )
    from patch_sim_ui.components.protocol_panel import _PROTOCOL_PARAM_SCHEMA

    expected = {(CURRENT_CLAMP, p) for p in CURRENT_PROTOCOLS} | {
        (VOLTAGE_CLAMP, p) for p in VOLTAGE_PROTOCOLS
    }
    assert set(_PROTOCOL_PARAM_SCHEMA.keys()) == expected


def test_protocol_param_schema_attrs_resolve_on_state():
    """Every schema ``attr`` must resolve to a ``ProtocolState`` field + setter.

    The data-driven builder calls ``getattr(ProtocolState, f.attr)`` and
    ``getattr(ProtocolState, f"set_{f.attr}")`` for each schema entry; this
    test catches typos or missing handlers at static-data time rather than at
    render time.
    """
    from patch_sim_ui.components.protocol_panel import _PROTOCOL_PARAM_SCHEMA
    from patch_sim_ui.state.protocol import ProtocolState

    for (clamp_mode, protocol_type), fields in _PROTOCOL_PARAM_SCHEMA.items():
        for field in fields:
            assert hasattr(ProtocolState, field.attr), (
                f"{clamp_mode}/{protocol_type}: ProtocolState has no attribute "
                f"{field.attr!r}"
            )
            assert hasattr(ProtocolState, f"set_{field.attr}"), (
                f"{clamp_mode}/{protocol_type}: ProtocolState has no setter "
                f"set_{field.attr!r}"
            )


def _protocol_schema_keys():
    """Return all (clamp_mode, protocol_type) keys from the schema.

    Used to parametrize the per-entry render test so the schema is the
    single source of truth — adding a new protocol entry automatically
    extends the test matrix.
    """
    from patch_sim_ui.components.protocol_panel import _PROTOCOL_PARAM_SCHEMA

    return list(_PROTOCOL_PARAM_SCHEMA.keys())


@pytest.mark.parametrize("clamp_mode,protocol_type", _protocol_schema_keys())
def test_build_param_form_renders_for_each_protocol(clamp_mode, protocol_type):
    """``_build_param_form`` must build without error for every schema entry."""
    from patch_sim_ui.components.protocol_panel import (
        _PROTOCOL_PARAM_SCHEMA,
        _build_param_form,
    )

    fields = _PROTOCOL_PARAM_SCHEMA[(clamp_mode, protocol_type)]
    _build_param_form(fields)


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
    from patch_sim_ui.components.metrics import analysis_sidebar

    analysis_sidebar()


def _spec_for_id(ui_id: str):
    """Return the ``_PER_CHANNEL_TRACE_SPECS`` entry for ``ui_id``."""
    from patch_sim_ui.channels import CHANNELS
    from patch_sim_ui.components.sweep_manager import _PER_CHANNEL_TRACE_SPECS

    idx = next(i for i, ch in enumerate(CHANNELS) if ch.id == ui_id)
    return _PER_CHANNEL_TRACE_SPECS[idx]


def test_channel_trace_group_excludes_current_when_flag_false():
    """_channel_trace_group with include_current=False must omit the current label.

    The current checkbox label (e.g. ``"I_Ih"``) must not appear in the rendered
    component while the gating label must still be present.
    """
    from patch_sim_ui.components.sweep_manager import _channel_trace_group

    *positional, has_gating = _spec_for_id("ih")
    current_label, gating_label = positional[2], positional[5]
    comp = _channel_trace_group(
        *positional, include_current=False, include_gating=has_gating
    )
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
    from patch_sim_ui.components.sweep_manager import _channel_trace_group

    *positional, has_gating = _spec_for_id("ih")
    current_label, gating_label = positional[2], positional[5]
    comp = _channel_trace_group(*positional, include_gating=has_gating)
    assert _label_present(comp, current_label), (
        f"Current label {current_label!r} must appear by default"
    )
    assert _label_present(comp, gating_label), (
        f"Gating label {gating_label!r} must appear by default"
    )


def test_channel_trace_group_skips_gating_when_no_gates():
    """Leaks (NaL, KL) have no gates — the gating checkbox must not appear."""
    from patch_sim_ui.components.sweep_manager import _channel_trace_group

    *positional, has_gating = _spec_for_id("nal")
    assert has_gating is False
    current_label, gating_label = positional[2], positional[5]
    comp = _channel_trace_group(*positional, include_gating=has_gating)
    assert _label_present(comp, current_label), (
        f"Current label {current_label!r} must appear for a current-only group"
    )
    assert not _label_present(comp, gating_label), (
        f"Gating label {gating_label!r} must not appear for a leak channel"
    )


def test_cc_per_channel_section_gating_only():
    """In CC mode each channel sub-group hides current and (if applicable) shows gating.

    For every channel the current label must be absent.  Channels with
    gating variables must show their gating label; leak channels must not.
    """
    from patch_sim_ui.channels import CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _PER_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for ch, (*positional, has_gating) in zip(CHANNELS, _PER_CHANNEL_TRACE_SPECS):
        comp = _channel_trace_group(
            *positional, include_current=False, include_gating=has_gating
        )
        assert not _label_present(comp, ch.current_label), (
            f"Channel {ch.label!r}: {ch.current_label!r} must be absent in CC mode"
        )
        if has_gating:
            assert _label_present(comp, ch.gating_label), (
                f"Channel {ch.label!r}: {ch.gating_label!r} must be present in CC mode"
            )


def test_vc_per_channel_section_includes_current():
    """In VC mode each channel sub-group shows current; gating only when present."""
    from patch_sim_ui.channels import CHANNELS
    from patch_sim_ui.components.sweep_manager import (
        _PER_CHANNEL_TRACE_SPECS,
        _channel_trace_group,
    )

    for ch, (*positional, has_gating) in zip(CHANNELS, _PER_CHANNEL_TRACE_SPECS):
        comp = _channel_trace_group(
            *positional, include_current=True, include_gating=has_gating
        )
        assert _label_present(comp, ch.current_label), (
            f"Channel {ch.label!r}: {ch.current_label!r} must be present in VC mode"
        )
        if has_gating:
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


def test_tau_v_plot_helper_renders_without_error():
    """The private _tau_v_plot helper instantiates without raising.

    The component is wrapped in an ``rx.cond`` against
    ``AnalysisState.has_tau_v_data`` so that it only appears once a τ-V
    analysis has been computed; this smoke-tests its construction.
    """
    from patch_sim_ui.components.metrics import _tau_v_plot

    _tau_v_plot()
