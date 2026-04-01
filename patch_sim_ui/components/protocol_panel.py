"""Protocol configuration panel component."""

import reflex as rx

from patch_sim.presets import PROTOCOL_PRESET_NAMES
from patch_sim_ui.constants import CURRENT_CLAMP, VOLTAGE_CLAMP
from patch_sim_ui.state import AppState


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
            AppState.pre_stimulus_duration,
            AppState.set_pre_stimulus_duration,
        ),
        _num_field(
            "Stimulus (ms)",
            AppState.stimulus_duration,
            AppState.set_stimulus_duration,
        ),
        _num_field(
            "Post-stimulus (ms)",
            AppState.post_stimulus_duration,
            AppState.set_post_stimulus_duration,
        ),
    )


def _cc_step_params() -> rx.Component:
    """Parameter fields for the current clamp Step protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Current min (µA/cm²)",
            AppState.min_stimulus,
            AppState.set_min_stimulus,
        ),
        _num_field(
            "Current max (µA/cm²)",
            AppState.max_stimulus,
            AppState.set_max_stimulus,
        ),
        _num_field(
            "Current step (µA/cm²)",
            AppState.stimulus_step,
            AppState.set_stimulus_step,
            disabled=AppState.is_step_single_sweep,
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
            AppState.start_current,
            AppState.set_start_current,
        ),
        _num_field(
            "End current (µA/cm²)", AppState.end_current, AppState.set_end_current
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
            AppState.pulse_amplitude,
            AppState.set_pulse_amplitude,
        ),
        _num_field("Pulse width (ms)", AppState.pulse_width, AppState.set_pulse_width),
        _num_field(
            "Pulse interval (ms)",
            AppState.pulse_interval,
            AppState.set_pulse_interval,
        ),
        spacing="2",
        width="100%",
    )


def _cc_sine_params() -> rx.Component:
    """Parameter fields for the current clamp Sinusoidal protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field("DC offset (µA/cm²)", AppState.dc_offset, AppState.set_dc_offset),
        _num_field("Amplitude (µA/cm²)", AppState.amplitude, AppState.set_amplitude),
        _num_field("Frequency (Hz)", AppState.frequency, AppState.set_frequency),
        spacing="2",
        width="100%",
    )


def _cc_chirp_params() -> rx.Component:
    """Parameter fields for the current clamp Chirp protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field("DC offset (µA/cm²)", AppState.dc_offset, AppState.set_dc_offset),
        _num_field("Amplitude (µA/cm²)", AppState.amplitude, AppState.set_amplitude),
        _num_field(
            "Start freq (Hz)", AppState.start_frequency, AppState.set_start_frequency
        ),
        _num_field("End freq (Hz)", AppState.end_frequency, AppState.set_end_frequency),
        spacing="2",
        width="100%",
    )


def _cc_noise_params() -> rx.Component:
    """Parameter fields for the current clamp Noise protocol."""
    return rx.vstack(
        *_duration_fields(),
        _num_field(
            "Mean current (µA/cm²)", AppState.mean_current, AppState.set_mean_current
        ),
        _num_field(
            "Std current (µA/cm²)", AppState.std_current, AppState.set_std_current
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
            AppState.min_stimulus,
            AppState.set_min_stimulus,
        ),
        _num_field(
            "Voltage max (mV)",
            AppState.max_stimulus,
            AppState.set_max_stimulus,
        ),
        _num_field(
            "Voltage step (mV)",
            AppState.stimulus_step,
            AppState.set_stimulus_step,
            disabled=AppState.is_step_single_sweep,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.holding_voltage,
            AppState.set_holding_voltage,
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
            AppState.vc_start_voltage,
            AppState.set_vc_start_voltage,
        ),
        _num_field(
            "End voltage (mV)", AppState.vc_end_voltage, AppState.set_vc_end_voltage
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.holding_voltage,
            AppState.set_holding_voltage,
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
            AppState.vc_pulse_amplitude,
            AppState.set_vc_pulse_amplitude,
        ),
        _num_field(
            "Pulse width (ms)", AppState.vc_pulse_width, AppState.set_vc_pulse_width
        ),
        _num_field(
            "Pulse interval (ms)",
            AppState.vc_pulse_interval,
            AppState.set_vc_pulse_interval,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.holding_voltage,
            AppState.set_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _current_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected current clamp protocol."""
    return rx.match(
        AppState.protocol_type,
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
        AppState.protocol_type,
        ("Step", _vc_step_params()),
        ("Ramp", _vc_ramp_params()),
        ("Pulse Train", _vc_pulse_params()),
        rx.fragment(),
    )


def protocol_panel() -> rx.Component:
    """Sidebar panel for experiment mode and protocol configuration."""
    return rx.vstack(
        rx.heading("Experiment", size="3"),
        rx.select(
            PROTOCOL_PRESET_NAMES,
            placeholder="Load preset…",
            on_change=AppState.load_protocol_preset,
            width="100%",
            size="2",
        ),
        rx.separator(),
        rx.text("Mode", size="2", weight="bold"),
        rx.radio_group(
            [CURRENT_CLAMP, VOLTAGE_CLAMP],
            value=AppState.clamp_mode,
            on_change=AppState.set_clamp_mode,
            direction="row",
        ),
        rx.separator(),
        rx.text("Protocol", size="2", weight="bold"),
        rx.select(
            AppState.protocol_options,
            value=AppState.protocol_type,
            on_change=AppState.set_protocol_type,
            width="100%",
        ),
        rx.cond(
            AppState.clamp_mode == CURRENT_CLAMP,
            _current_protocol_params(),
            _voltage_protocol_params(),
        ),
        spacing="3",
        width="100%",
        padding="4",
    )
