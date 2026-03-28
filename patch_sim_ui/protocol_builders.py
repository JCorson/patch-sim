"""Pure functions for building simulation protocol arrays.

These functions are extracted from AppState to allow unit testing without
the Reflex runtime.
"""

import logging

import numpy as np

import patch_sim

logger = logging.getLogger(__name__)


def build_current_protocol(
    protocol_type: str,
    sampling_frequency: float,
    pre_stimulus_duration: float = 10.0,
    stimulus_duration: float = 30.0,
    post_stimulus_duration: float = 10.0,
    current_amplitude: float = 10.0,
    start_current: float = 0.0,
    end_current: float = 15.0,
    pulse_amplitude: float = 10.0,
    pulse_width: float = 2.0,
    pulse_interval: float = 10.0,
    dc_offset: float = 8.0,
    amplitude: float = 4.0,
    frequency: float = 50.0,
    start_frequency: float = 1.0,
    end_frequency: float = 100.0,
    mean_current: float = 8.0,
    std_current: float = 2.0,
) -> list[tuple[np.ndarray, str]]:
    """Build current clamp protocol arrays from explicit parameters.

    Args:
        protocol_type: One of "Step", "Ramp", "Pulse Train", "Sinusoidal",
            "Chirp", or "Noise".
        sampling_frequency: Sampling frequency in Hz.
        pre_stimulus_duration: Duration before the stimulus in ms.
        stimulus_duration: Duration of the stimulus in ms.
        post_stimulus_duration: Duration after the stimulus in ms.
        current_amplitude: Step current amplitude in µA/cm².
        start_current: Ramp start current in µA/cm².
        end_current: Ramp end current in µA/cm².
        pulse_amplitude: Pulse amplitude in µA/cm².
        pulse_width: Pulse width in ms.
        pulse_interval: Interval between pulse starts in ms.
        dc_offset: DC offset for sinusoidal/chirp in µA/cm².
        amplitude: AC amplitude for sinusoidal/chirp in µA/cm².
        frequency: Sinusoidal frequency in Hz.
        start_frequency: Chirp start frequency in Hz.
        end_frequency: Chirp end frequency in Hz.
        mean_current: Noise mean current in µA/cm².
        std_current: Noise standard deviation in µA/cm².

    Returns:
        List of (stimulus_array, sweep_label) pairs. Current clamp protocols
        always return a single-element list with an empty label.

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    total_duration = pre_stimulus_duration + stimulus_duration + post_stimulus_duration
    if protocol_type == "Step":
        protocol = patch_sim.step_current(
            duration=total_duration,
            current_amplitude=current_amplitude,
            step_start=pre_stimulus_duration,
            step_duration=stimulus_duration,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Ramp":
        protocol = patch_sim.ramp_current(
            duration=total_duration,
            start_current=start_current,
            end_current=end_current,
            ramp_start=pre_stimulus_duration,
            ramp_duration=stimulus_duration,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Pulse Train":
        protocol = patch_sim.pulse_train(
            duration=total_duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=pre_stimulus_duration,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Sinusoidal":
        protocol = patch_sim.sinusoidal_current(
            duration=total_duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Chirp":
        protocol = patch_sim.chirp_current(
            duration=total_duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            start_frequency=start_frequency,
            end_frequency=end_frequency,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Noise":
        protocol = patch_sim.noise_current(
            duration=total_duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )
    else:
        raise ValueError(f"Unknown current protocol type: {protocol_type!r}")
    logger.debug(
        "build_current_protocol: type=%r total_duration=%.1f ms steps=%d",
        protocol_type,
        total_duration,
        len(protocol),
    )
    return [(protocol, "")]


def build_voltage_protocol(
    protocol_type: str,
    sampling_frequency: float,
    pre_stimulus_duration: float = 10.0,
    stimulus_duration: float = 30.0,
    post_stimulus_duration: float = 10.0,
    vc_holding_voltage: float = -70.0,
    vc_voltage_amplitude: float = 0.0,
    vc_start_voltage: float = -70.0,
    vc_end_voltage: float = 40.0,
    vc_pulse_amplitude: float = 20.0,
    vc_pulse_width: float = 2.0,
    vc_pulse_interval: float = 10.0,
    vc_voltage_min: float = -100.0,
    vc_voltage_max: float = 60.0,
    vc_voltage_step: float = 10.0,
) -> list[tuple[np.ndarray, str]]:
    """Build voltage clamp protocol arrays from explicit parameters.

    Args:
        protocol_type: One of "Step", "Ramp", "Pulse Train", or "I-V Curve".
        sampling_frequency: Sampling frequency in Hz.
        pre_stimulus_duration: Duration before the stimulus in ms.
        stimulus_duration: Duration of the stimulus in ms.
        post_stimulus_duration: Duration after the stimulus in ms.
        vc_holding_voltage: Holding voltage in mV.
        vc_voltage_amplitude: Step voltage amplitude in mV.
        vc_start_voltage: Ramp start voltage in mV.
        vc_end_voltage: Ramp end voltage in mV.
        vc_pulse_amplitude: Pulse amplitude in mV.
        vc_pulse_width: Pulse width in ms.
        vc_pulse_interval: Interval between pulse starts in ms.
        vc_voltage_min: Minimum voltage for I-V curve in mV.
        vc_voltage_max: Maximum voltage for I-V curve in mV.
        vc_voltage_step: Voltage step size in mV (I-V curve).

    Returns:
        List of (stimulus_array, sweep_label) pairs. Single-sweep protocols
        return a one-element list with an empty label. I-V Curve returns one
        entry per voltage step labelled "+XX mV" / "-XX mV".

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    total_duration = pre_stimulus_duration + stimulus_duration + post_stimulus_duration
    result: list[tuple[np.ndarray, str]]
    if protocol_type == "Step":
        protocol = patch_sim.step_voltage(
            duration=total_duration,
            voltage_amplitude=vc_voltage_amplitude,
            step_start=pre_stimulus_duration,
            step_duration=stimulus_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
        result = [(protocol, "")]
    elif protocol_type == "Ramp":
        protocol = patch_sim.ramp_voltage(
            duration=total_duration,
            start_voltage=vc_start_voltage,
            end_voltage=vc_end_voltage,
            ramp_start=pre_stimulus_duration,
            ramp_duration=stimulus_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
        result = [(protocol, "")]
    elif protocol_type == "Pulse Train":
        protocol = patch_sim.pulse_train_voltage(
            duration=total_duration,
            pulse_amplitude=vc_pulse_amplitude,
            pulse_width=vc_pulse_width,
            pulse_interval=vc_pulse_interval,
            train_start=pre_stimulus_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
        result = [(protocol, "")]
    elif protocol_type == "I-V Curve":
        voltage_range = vc_voltage_max - vc_voltage_min
        n_steps = round(voltage_range / vc_voltage_step) + 1
        voltages = np.linspace(vc_voltage_min, vc_voltage_max, n_steps)
        result = [
            (
                patch_sim.step_voltage(
                    duration=total_duration,
                    voltage_amplitude=float(voltage),
                    step_start=pre_stimulus_duration,
                    step_duration=stimulus_duration,
                    holding_voltage=vc_holding_voltage,
                    sampling_frequency=sampling_frequency,
                ),
                f"{voltage:+.0f} mV",
            )
            for voltage in voltages
        ]
    else:
        raise ValueError(f"Unknown voltage protocol type: {protocol_type!r}")
    logger.debug(
        "build_voltage_protocol: type=%r total_duration=%.1f ms sweeps=%d",
        protocol_type,
        total_duration,
        len(result),
    )
    return result
