"""Protocol configuration panel component."""

from dataclasses import dataclass

import reflex as rx

from patch_sim.presets import PROTOCOL_PRESET_NAMES
from patch_sim_ui.constants import (
    CURRENT_CLAMP,
    CURRENT_PROTOCOLS,
    VOLTAGE_CLAMP,
    VOLTAGE_PROTOCOLS,
)
from patch_sim_ui.state.protocol import ProtocolState


def _num_field(
    label: str,
    var: rx.Var,
    handler,
    unit: str = "",
    disabled: rx.Var | bool = False,
) -> rx.Component:
    """Render a labeled numeric input field.

    Args:
        label: Display label shown to the left of the input.
        var: Reactive variable bound to the input value.
        handler: Event handler called on change.
        unit: Optional unit label shown to the right of the input.
        disabled: When True (or a reactive bool that is True), the input is
            rendered in a disabled state and cannot be edited.
    """
    return rx.hstack(
        rx.text(label, size="2", color="gray", width="160px"),
        rx.input(
            value=var,
            on_change=handler,
            width="100px",
            size="1",
            type="number",
            disabled=disabled,
        ),
        rx.text(unit, size="1", color="gray") if unit else rx.fragment(),
        width="100%",
        spacing="2",
        align="center",
    )


def _duration_fields() -> tuple[rx.Component, rx.Component, rx.Component]:
    """Return the three shared pre/stimulus/post-stimulus duration fields."""
    return (
        _num_field(
            "Pre-stimulus (ms)",
            ProtocolState.pre_stimulus_duration,
            ProtocolState.set_pre_stimulus_duration,
        ),
        _num_field(
            "Stimulus (ms)",
            ProtocolState.stimulus_duration,
            ProtocolState.set_stimulus_duration,
        ),
        _num_field(
            "Post-stimulus (ms)",
            ProtocolState.post_stimulus_duration,
            ProtocolState.set_post_stimulus_duration,
        ),
    )


@dataclass(frozen=True)
class ParamField:
    """Schema entry describing one numeric parameter input.

    Attributes:
        label: Display label (units inlined, e.g. ``"Current min (µA/cm²)"``).
        attr: ``ProtocolState`` attribute name; the matching event handler is
            looked up as ``set_<attr>`` unless ``setter_attr`` overrides it.
        disabled: Optional reactive flag (or static bool) forwarded to the
            input's ``disabled`` prop.
        setter_attr: Optional explicit setter name on ``ProtocolState``.  When
            ``None`` (default) the setter is derived as ``set_<attr>``.
    """

    label: str
    attr: str
    disabled: rx.Var | bool = False
    setter_attr: str | None = None


_PROTOCOL_PARAM_SCHEMA: dict[tuple[str, str], tuple[ParamField, ...]] = {
    (CURRENT_CLAMP, "Step"): (
        ParamField("Current min (µA/cm²)", "min_stimulus"),
        ParamField("Current max (µA/cm²)", "max_stimulus"),
        ParamField("Current step (µA/cm²)", "stimulus_step"),
    ),
    (CURRENT_CLAMP, "Ramp"): (
        ParamField("Start current (µA/cm²)", "start_current"),
        ParamField("End current (µA/cm²)", "end_current"),
    ),
    (CURRENT_CLAMP, "Pulse Train"): (
        ParamField("Pulse amplitude (µA/cm²)", "pulse_amplitude"),
        ParamField("Pulse width (ms)", "pulse_width"),
        ParamField("Pulse interval (ms)", "pulse_interval"),
    ),
    (CURRENT_CLAMP, "Sinusoidal"): (
        ParamField("DC offset (µA/cm²)", "dc_offset"),
        ParamField("Amplitude (µA/cm²)", "amplitude"),
        ParamField("Frequency (Hz)", "frequency"),
    ),
    (CURRENT_CLAMP, "Chirp"): (
        ParamField("DC offset (µA/cm²)", "dc_offset"),
        ParamField("Amplitude (µA/cm²)", "amplitude"),
        ParamField("Start freq (Hz)", "start_frequency"),
        ParamField("End freq (Hz)", "end_frequency"),
    ),
    (CURRENT_CLAMP, "Noise"): (
        ParamField("Mean current (µA/cm²)", "mean_current"),
        ParamField("Std current (µA/cm²)", "std_current"),
    ),
    (VOLTAGE_CLAMP, "Step"): (
        ParamField("Voltage min (mV)", "min_stimulus"),
        ParamField("Voltage max (mV)", "max_stimulus"),
        ParamField("Voltage step (mV)", "stimulus_step"),
        ParamField("Holding voltage (mV)", "holding_voltage"),
    ),
    (VOLTAGE_CLAMP, "Ramp"): (
        ParamField("Start voltage (mV)", "vc_start_voltage"),
        ParamField("End voltage (mV)", "vc_end_voltage"),
        ParamField("Holding voltage (mV)", "holding_voltage"),
    ),
    (VOLTAGE_CLAMP, "Pulse Train"): (
        ParamField("Pulse amplitude (mV)", "vc_pulse_amplitude"),
        ParamField("Pulse width (ms)", "vc_pulse_width"),
        ParamField("Pulse interval (ms)", "vc_pulse_interval"),
        ParamField("Holding voltage (mV)", "holding_voltage"),
    ),
    # Two-pulse steady-state-inactivation protocol.  The shared duration fields
    # double as the conditioning-prepulse duration ("Pre-stimulus") and the
    # test-pulse duration ("Stimulus"); the min/max/step trio sweeps the
    # conditioning prepulse voltage and "Test pulse" is the fixed test voltage.
    (VOLTAGE_CLAMP, "Inactivation"): (
        ParamField("Prepulse min (mV)", "min_stimulus"),
        ParamField("Prepulse max (mV)", "max_stimulus"),
        ParamField("Prepulse step (mV)", "stimulus_step"),
        ParamField("Test pulse (mV)", "test_pulse_voltage"),
    ),
}

# Single-sweep amplitude fields for the Step protocol.  In single-sweep mode
# the user edits one combined value and ``set_single_stimulus`` mirrors it
# into both ``min_stimulus`` and ``max_stimulus`` so the existing builders
# emit one sweep.  Voltage clamp keeps its holding-voltage row alongside.
_STEP_SINGLE_FIELDS: dict[str, tuple[ParamField, ...]] = {
    CURRENT_CLAMP: (
        ParamField(
            "Current (µA/cm²)", "min_stimulus", setter_attr="set_single_stimulus"
        ),
    ),
    VOLTAGE_CLAMP: (
        ParamField("Voltage (mV)", "min_stimulus", setter_attr="set_single_stimulus"),
        ParamField("Holding voltage (mV)", "holding_voltage"),
    ),
}


def _render_field(f: ParamField) -> rx.Component:
    """Render a single ParamField as a numeric input row."""
    return _num_field(
        f.label,
        getattr(ProtocolState, f.attr),
        getattr(ProtocolState, f.setter_attr or f"set_{f.attr}"),
        disabled=f.disabled,
    )


def _build_param_form(fields: tuple[ParamField, ...]) -> rx.Component:
    """Render the shared duration fields followed by ``fields``.

    Args:
        fields: Schema entries describing the protocol-specific numeric inputs.

    Returns:
        Stacked component containing duration inputs and one ``_num_field``
        per schema entry.
    """
    return rx.vstack(
        *_duration_fields(),
        *[_render_field(f) for f in fields],
        spacing="2",
        width="100%",
    )


def _sweep_mode_toggle() -> rx.Component:
    """Multi-sweep toggle switch for the Step protocol.

    Off means single sweep (one combined amplitude field); on means
    multi-sweep (min/max/step trio).  Driven by ``is_step_single_sweep``
    so external edits that drive ``min == max`` flip the switch off too.
    """
    return rx.hstack(
        rx.text("Multi-sweep", size="2", color="gray", width="160px"),
        rx.switch(
            checked=~ProtocolState.is_step_single_sweep,
            on_change=ProtocolState.set_step_multi_sweep,
        ),
        width="100%",
        spacing="2",
        align="center",
    )


def _step_form(clamp_mode: str) -> rx.Component:
    """Step protocol form with sweep-mode toggle and mode-swapped fields.

    In single-sweep mode the form shows one combined amplitude field; in
    multi-sweep mode it shows the standard min/max/step trio (and, for
    voltage clamp, the holding-voltage row).  The toggle drives the swap
    via ``ProtocolState.is_step_single_sweep``.

    Args:
        clamp_mode: ``CURRENT_CLAMP`` or ``VOLTAGE_CLAMP``.
    """
    multi_fields = _PROTOCOL_PARAM_SCHEMA[(clamp_mode, "Step")]
    single_fields = _STEP_SINGLE_FIELDS[clamp_mode]
    return rx.vstack(
        *_duration_fields(),
        _sweep_mode_toggle(),
        rx.cond(
            ProtocolState.is_step_single_sweep,
            rx.vstack(
                *[_render_field(f) for f in single_fields],
                spacing="2",
                width="100%",
            ),
            rx.vstack(
                *[_render_field(f) for f in multi_fields],
                spacing="2",
                width="100%",
            ),
        ),
        spacing="2",
        width="100%",
    )


def _current_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected current clamp protocol."""
    return rx.match(
        ProtocolState.protocol_type,
        ("Step", _step_form(CURRENT_CLAMP)),
        *[
            (proto, _build_param_form(_PROTOCOL_PARAM_SCHEMA[(CURRENT_CLAMP, proto)]))
            for proto in CURRENT_PROTOCOLS
            if proto != "Step"
        ],
        rx.fragment(),
    )


def _voltage_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected voltage clamp protocol."""
    return rx.match(
        ProtocolState.protocol_type,
        ("Step", _step_form(VOLTAGE_CLAMP)),
        *[
            (proto, _build_param_form(_PROTOCOL_PARAM_SCHEMA[(VOLTAGE_CLAMP, proto)]))
            for proto in VOLTAGE_PROTOCOLS
            if proto != "Step"
        ],
        rx.fragment(),
    )


def protocol_panel() -> rx.Component:
    """Sidebar panel for experiment mode and protocol configuration."""
    return rx.vstack(
        rx.heading("Experiment", size="4"),
        rx.select(
            PROTOCOL_PRESET_NAMES,
            placeholder="Load preset…",
            value=ProtocolState.active_protocol_preset,
            on_change=ProtocolState.load_protocol_preset,
            width="100%",
            size="2",
        ),
        rx.separator(),
        rx.text("Mode", size="2", weight="bold"),
        rx.radio_group(
            [CURRENT_CLAMP, VOLTAGE_CLAMP],
            value=ProtocolState.clamp_mode,
            on_change=ProtocolState.set_clamp_mode,
            direction="row",
        ),
        rx.separator(),
        rx.text("Protocol", size="2", weight="bold"),
        rx.select(
            ProtocolState.protocol_options,
            value=ProtocolState.protocol_type,
            on_change=ProtocolState.set_protocol_type,
            width="100%",
        ),
        rx.cond(
            ProtocolState.clamp_mode == CURRENT_CLAMP,
            _current_protocol_params(),
            _voltage_protocol_params(),
        ),
        spacing="3",
        width="100%",
        padding="4",
    )
