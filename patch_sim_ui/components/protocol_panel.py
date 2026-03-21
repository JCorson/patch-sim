"""Protocol configuration panel component."""

import reflex as rx

from patch_sim_ui.state import AppState


def _num_field(label: str, var: rx.Var, handler, unit: str = "") -> rx.Component:
    """Render a labelled numeric input field."""
    return rx.hstack(
        rx.text(label, size="2", color="gray", width="160px"),
        rx.input(
            value=var,
            on_change=handler,
            width="100px",
            size="1",
            type="number",
        ),
        rx.text(unit, size="1", color="gray") if unit else rx.fragment(),
        width="100%",
        spacing="2",
        align="center",
    )


def _cc_step_params() -> rx.Component:
    """Parameter fields for the current clamp Step protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Amplitude (µA/cm²)",
            AppState.current_amplitude,
            AppState.set_current_amplitude,
        ),
        _num_field("Step start (ms)", AppState.step_start, AppState.set_step_start),
        _num_field(
            "Step duration (ms)", AppState.step_duration, AppState.set_step_duration
        ),
        spacing="2",
        width="100%",
    )


def _cc_ramp_params() -> rx.Component:
    """Parameter fields for the current clamp Ramp protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Start current (µA/cm²)",
            AppState.start_current,
            AppState.set_start_current,
        ),
        _num_field(
            "End current (µA/cm²)", AppState.end_current, AppState.set_end_current
        ),
        _num_field("Ramp start (ms)", AppState.ramp_start, AppState.set_ramp_start),
        _num_field(
            "Ramp duration (ms)", AppState.ramp_duration, AppState.set_ramp_duration
        ),
        spacing="2",
        width="100%",
    )


def _cc_pulse_params() -> rx.Component:
    """Parameter fields for the current clamp Pulse Train protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
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
        _num_field("Train start (ms)", AppState.train_start, AppState.set_train_start),
        spacing="2",
        width="100%",
    )


def _cc_sine_params() -> rx.Component:
    """Parameter fields for the current clamp Sinusoidal protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
        _num_field("DC offset (µA/cm²)", AppState.dc_offset, AppState.set_dc_offset),
        _num_field("Amplitude (µA/cm²)", AppState.amplitude, AppState.set_amplitude),
        _num_field("Frequency (Hz)", AppState.frequency, AppState.set_frequency),
        spacing="2",
        width="100%",
    )


def _cc_chirp_params() -> rx.Component:
    """Parameter fields for the current clamp Chirp protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
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
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
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
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Voltage amplitude (mV)",
            AppState.vc_voltage_amplitude,
            AppState.set_vc_voltage_amplitude,
        ),
        _num_field(
            "Step start (ms)", AppState.vc_step_start, AppState.set_vc_step_start
        ),
        _num_field(
            "Step duration (ms)",
            AppState.vc_step_duration,
            AppState.set_vc_step_duration,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.vc_holding_voltage,
            AppState.set_vc_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_ramp_params() -> rx.Component:
    """Parameter fields for the voltage clamp Ramp protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Start voltage (mV)",
            AppState.vc_start_voltage,
            AppState.set_vc_start_voltage,
        ),
        _num_field(
            "End voltage (mV)", AppState.vc_end_voltage, AppState.set_vc_end_voltage
        ),
        _num_field(
            "Ramp start (ms)", AppState.vc_ramp_start, AppState.set_vc_ramp_start
        ),
        _num_field(
            "Ramp duration (ms)",
            AppState.vc_ramp_duration,
            AppState.set_vc_ramp_duration,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.vc_holding_voltage,
            AppState.set_vc_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_pulse_params() -> rx.Component:
    """Parameter fields for the voltage clamp Pulse Train protocol."""
    return rx.vstack(
        _num_field("Duration (ms)", AppState.duration, AppState.set_duration),
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
            "Train start (ms)", AppState.vc_train_start, AppState.set_vc_train_start
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.vc_holding_voltage,
            AppState.set_vc_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_iv_params() -> rx.Component:
    """Parameter fields for the voltage clamp I-V Curve protocol."""
    return rx.vstack(
        _num_field("Step duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Voltage min (mV)",
            AppState.vc_voltage_min,
            AppState.set_vc_voltage_min,
        ),
        _num_field(
            "Voltage max (mV)",
            AppState.vc_voltage_max,
            AppState.set_vc_voltage_max,
        ),
        _num_field(
            "Voltage step (mV)",
            AppState.vc_voltage_step,
            AppState.set_vc_voltage_step,
        ),
        _num_field(
            "Pre-pulse (ms)",
            AppState.vc_pre_pulse_duration,
            AppState.set_vc_pre_pulse_duration,
        ),
        _num_field(
            "Post-pulse (ms)",
            AppState.vc_post_pulse_duration,
            AppState.set_vc_post_pulse_duration,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.vc_holding_voltage,
            AppState.set_vc_holding_voltage,
        ),
        spacing="2",
        width="100%",
    )


def _vc_activation_params() -> rx.Component:
    """Parameter fields for the voltage clamp Activation protocol."""
    return rx.vstack(
        _num_field("Test duration (ms)", AppState.duration, AppState.set_duration),
        _num_field(
            "Prepulse voltage (mV)",
            AppState.vc_prepulse_voltage,
            AppState.set_vc_prepulse_voltage,
        ),
        _num_field(
            "Prepulse duration (ms)",
            AppState.vc_prepulse_duration,
            AppState.set_vc_prepulse_duration,
        ),
        _num_field(
            "Test V min (mV)",
            AppState.vc_test_voltage_min,
            AppState.set_vc_test_voltage_min,
        ),
        _num_field(
            "Test V max (mV)",
            AppState.vc_test_voltage_max,
            AppState.set_vc_test_voltage_max,
        ),
        _num_field(
            "Voltage step (mV)",
            AppState.vc_voltage_step,
            AppState.set_vc_voltage_step,
        ),
        _num_field(
            "Interpulse (ms)",
            AppState.vc_interpulse_duration,
            AppState.set_vc_interpulse_duration,
        ),
        _num_field(
            "Holding voltage (mV)",
            AppState.vc_holding_voltage,
            AppState.set_vc_holding_voltage,
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
        _cc_noise_params(),  # default: Noise
    )


def _voltage_protocol_params() -> rx.Component:
    """Dynamic parameter form for the selected voltage clamp protocol."""
    return rx.match(
        AppState.protocol_type,
        ("Step", _vc_step_params()),
        ("Ramp", _vc_ramp_params()),
        ("Pulse Train", _vc_pulse_params()),
        ("I-V Curve", _vc_iv_params()),
        _vc_activation_params(),  # default: Activation
    )


def protocol_panel() -> rx.Component:
    """Sidebar panel for experiment mode and protocol configuration."""
    return rx.vstack(
        rx.heading("Experiment", size="3"),
        rx.separator(),
        rx.text("Mode", size="2", weight="bold"),
        rx.radio_group(
            ["Current Clamp", "Voltage Clamp"],
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
            AppState.clamp_mode == "Current Clamp",
            _current_protocol_params(),
            _voltage_protocol_params(),
        ),
        spacing="3",
        width="100%",
        padding="4",
    )
