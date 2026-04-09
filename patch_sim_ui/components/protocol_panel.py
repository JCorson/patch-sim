"""Protocol configuration panel component."""

import reflex as rx

from patch_sim.presets import PROTOCOL_PRESET_NAMES
from patch_sim_ui.constants import CURRENT_CLAMP, VOLTAGE_CLAMP
from patch_sim_ui.state.protocol import ProtocolState


def _num_field(
    label: str,
    var: rx.Var,
    handler,
    unit: str = "",
    disabled: rx.Var | bool = False,
) -> rx.Component:
    """Render a labelled numeric input field.

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


def _cc_step_params() -> rx.Component:
    """Parameter fields for the current clamp Step protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Current min (µA/cm²)",
            ProtocolState.min_stimulus,
            ProtocolState.set_min_stimulus,
        ),
        _num_field(
            "Current max (µA/cm²)",
            ProtocolState.max_stimulus,
            ProtocolState.set_max_stimulus,
        ),
        _num_field(
            "Current step (µA/cm²)",
            ProtocolState.stimulus_step,
            ProtocolState.set_stimulus_step,
            disabled=ProtocolState.is_step_single_sweep,
        ),
        spacing="2",
        width="100%",
    )


def _cc_ramp_params() -> rx.Component:
    """Parameter fields for the current clamp Ramp protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Start current (µA/cm²)",
            ProtocolState.start_current,
            ProtocolState.set_start_current,
        ),
        _num_field(
            "End current (µA/cm²)",
            ProtocolState.end_current,
            ProtocolState.set_end_current,
        ),
        spacing="2",
        width="100%",
    )


def _cc_pulse_params() -> rx.Component:
    """Parameter fields for the current clamp Pulse Train protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Pulse amplitude (µA/cm²)",
            ProtocolState.pulse_amplitude,
            ProtocolState.set_pulse_amplitude,
        ),
        _num_field(
            "Pulse width (ms)", ProtocolState.pulse_width, ProtocolState.set_pulse_width
        ),
        _num_field(
            "Pulse interval (ms)",
            ProtocolState.pulse_interval,
            ProtocolState.set_pulse_interval,
        ),
        spacing="2",
        width="100%",
    )


def _cc_sine_params() -> rx.Component:
    """Parameter fields for the current clamp Sinusoidal protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "DC offset (µA/cm²)", ProtocolState.dc_offset, ProtocolState.set_dc_offset
        ),
        _num_field(
            "Amplitude (µA/cm²)", ProtocolState.amplitude, ProtocolState.set_amplitude
        ),
        _num_field(
            "Frequency (Hz)", ProtocolState.frequency, ProtocolState.set_frequency
        ),
        spacing="2",
        width="100%",
    )


def _cc_chirp_params() -> rx.Component:
    """Parameter fields for the current clamp Chirp protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "DC offset (µA/cm²)", ProtocolState.dc_offset, ProtocolState.set_dc_offset
        ),
        _num_field(
            "Amplitude (µA/cm²)", ProtocolState.amplitude, ProtocolState.set_amplitude
        ),
        _num_field(
            "Start freq (Hz)",
            ProtocolState.start_frequency,
            ProtocolState.set_start_frequency,
        ),
        _num_field(
            "End freq (Hz)",
            ProtocolState.end_frequency,
            ProtocolState.set_end_frequency,
        ),
        spacing="2",
        width="100%",
    )


def _cc_noise_params() -> rx.Component:
    """Parameter fields for the current clamp Noise protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Mean current (µA/cm²)",
            ProtocolState.mean_current,
            ProtocolState.set_mean_current,
        ),
        _num_field(
            "Std current (µA/cm²)",
            ProtocolState.std_current,
            ProtocolState.set_std_current,
        ),
        spacing="2",
        width="100%",
    )


def _vc_step_params() -> rx.Component:
    """Parameter fields for the voltage clamp Step protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Voltage min (mV)",
            ProtocolState.min_stimulus,
            ProtocolState.set_min_stimulus,
        ),
        _num_field(
            "Voltage max (mV)",
            ProtocolState.max_stimulus,
            ProtocolState.set_max_stimulus,
        ),
        _num_field(
            "Voltage step (mV)",
            ProtocolState.stimulus_step,
            ProtocolState.set_stimulus_step,
            disabled=ProtocolState.is_step_single_sweep,
        ),
        _num_field(
            "Holding voltage (mV)",
            ProtocolState.holding_voltage,
            ProtocolState.set_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_ramp_params() -> rx.Component:
    """Parameter fields for the voltage clamp Ramp protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Start voltage (mV)",
            ProtocolState.vc_start_voltage,
            ProtocolState.set_vc_start_voltage,
        ),
        _num_field(
            "End voltage (mV)",
            ProtocolState.vc_end_voltage,
            ProtocolState.set_vc_end_voltage,
        ),
        _num_field(
            "Holding voltage (mV)",
            ProtocolState.holding_voltage,
            ProtocolState.set_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_pulse_params() -> rx.Component:
    """Parameter fields for the voltage clamp Pulse Train protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Pulse amplitude (mV)",
            ProtocolState.vc_pulse_amplitude,
            ProtocolState.set_vc_pulse_amplitude,
        ),
        _num_field(
            "Pulse width (ms)",
            ProtocolState.vc_pulse_width,
            ProtocolState.set_vc_pulse_width,
        ),
        _num_field(
            "Pulse interval (ms)",
            ProtocolState.vc_pulse_interval,
            ProtocolState.set_vc_pulse_interval,
        ),
        _num_field(
            "Holding voltage (mV)",
            ProtocolState.holding_voltage,
            ProtocolState.set_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _current_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected current clamp protocol."""
    return rx.match(
        ProtocolState.protocol_type,
        ("Step", _cc_step_params()),
        ("Ramp", _cc_ramp_params()),
        ("Pulse Train", _cc_pulse_params()),
        ("Sinusoidal", _cc_sine_params()),
        ("Chirp", _cc_chirp_params()),
        ("Noise", _cc_noise_params()),
        rx.fragment(),
    )


def _voltage_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected voltage clamp protocol."""
    return rx.match(
        ProtocolState.protocol_type,
        ("Step", _vc_step_params()),
        ("Ramp", _vc_ramp_params()),
        ("Pulse Train", _vc_pulse_params()),
        rx.fragment(),
    )


def protocol_panel() -> rx.Component:
    """Sidebar panel for experiment mode and protocol configuration."""
    return rx.vstack(
        rx.heading("Experiment", size="4"),
        rx.select(
            PROTOCOL_PRESET_NAMES,
            placeholder="Load preset…",
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
