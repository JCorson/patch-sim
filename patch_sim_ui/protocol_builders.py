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
    duration: float,
    sampling_frequency: float,
    current_amplitude: float = 10.0,
    step_start: float = 10.0,
    step_duration: float = 30.0,
    start_current: float = 0.0,
    end_current: float = 15.0,
    ramp_start: float = 0.0,
    ramp_duration: float = 40.0,
    pulse_amplitude: float = 10.0,
    pulse_width: float = 2.0,
    pulse_interval: float = 10.0,
    train_start: float = 5.0,
    dc_offset: float = 8.0,
    amplitude: float = 4.0,
    frequency: float = 50.0,
    start_frequency: float = 1.0,
    end_frequency: float = 100.0,
    mean_current: float = 8.0,
    std_current: float = 2.0,
) -> np.ndarray:
    """Build a current clamp protocol array from explicit parameters.

    Args:
        protocol_type: One of "Step", "Ramp", "Pulse Train", "Sinusoidal",
            "Chirp", or "Noise".
        duration: Total duration in ms.
        sampling_frequency: Sampling frequency in Hz.
        current_amplitude: Step current amplitude in µA/cm².
        step_start: Step onset time in ms.
        step_duration: Step duration in ms.
        start_current: Ramp start current in µA/cm².
        end_current: Ramp end current in µA/cm².
        ramp_start: Ramp onset time in ms.
        ramp_duration: Ramp duration in ms.
        pulse_amplitude: Pulse amplitude in µA/cm².
        pulse_width: Pulse width in ms.
        pulse_interval: Interval between pulse starts in ms.
        train_start: Train onset time in ms.
        dc_offset: DC offset for sinusoidal/chirp in µA/cm².
        amplitude: AC amplitude for sinusoidal/chirp in µA/cm².
        frequency: Sinusoidal frequency in Hz.
        start_frequency: Chirp start frequency in Hz.
        end_frequency: Chirp end frequency in Hz.
        mean_current: Noise mean current in µA/cm².
        std_current: Noise standard deviation in µA/cm².

    Returns:
        Protocol array in µA/cm².

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    if protocol_type == "Step":
        protocol = patch_sim.step_current(
            duration=duration,
            current_amplitude=current_amplitude,
            step_start=step_start,
            step_duration=step_duration,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Ramp":
        protocol = patch_sim.ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Pulse Train":
        protocol = patch_sim.pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=train_start,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Sinusoidal":
        protocol = patch_sim.sinusoidal_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Chirp":
        protocol = patch_sim.chirp_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            start_frequency=start_frequency,
            end_frequency=end_frequency,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Noise":
        protocol = patch_sim.noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )
    else:
        raise ValueError(f"Unknown current protocol type: {protocol_type!r}")
    logger.debug(
        "build_current_protocol: type=%r duration=%.1f ms steps=%d",
        protocol_type,
        duration,
        len(protocol),
    )
    return protocol


def build_voltage_protocol(
    protocol_type: str,
    duration: float,
    sampling_frequency: float,
    vc_holding_voltage: float = -70.0,
    vc_voltage_amplitude: float = 0.0,
    vc_step_start: float = 10.0,
    vc_step_duration: float = 30.0,
    vc_start_voltage: float = -70.0,
    vc_end_voltage: float = 40.0,
    vc_ramp_start: float = 0.0,
    vc_ramp_duration: float = 40.0,
    vc_pulse_amplitude: float = 20.0,
    vc_pulse_width: float = 2.0,
    vc_pulse_interval: float = 10.0,
    vc_train_start: float = 5.0,
    vc_voltage_min: float = -100.0,
    vc_voltage_max: float = 60.0,
    vc_voltage_step: float = 10.0,
    vc_pre_pulse_duration: float = 5.0,
    vc_post_pulse_duration: float = 5.0,
) -> np.ndarray:
    """Build a voltage clamp protocol array from explicit parameters.

    Args:
        protocol_type: One of "Step", "Ramp", "Pulse Train", or "I-V Curve".
        duration: Total duration in ms (or step duration for I-V).
        sampling_frequency: Sampling frequency in Hz.
        vc_holding_voltage: Holding voltage in mV.
        vc_voltage_amplitude: Step voltage amplitude in mV.
        vc_step_start: Step onset time in ms.
        vc_step_duration: Step duration in ms.
        vc_start_voltage: Ramp start voltage in mV.
        vc_end_voltage: Ramp end voltage in mV.
        vc_ramp_start: Ramp onset time in ms.
        vc_ramp_duration: Ramp duration in ms.
        vc_pulse_amplitude: Pulse amplitude in mV.
        vc_pulse_width: Pulse width in ms.
        vc_pulse_interval: Interval between pulse starts in ms.
        vc_train_start: Train onset time in ms.
        vc_voltage_min: Minimum voltage for I-V curve in mV.
        vc_voltage_max: Maximum voltage for I-V curve in mV.
        vc_voltage_step: Voltage step size in mV (I-V curve).
        vc_pre_pulse_duration: Pre-pulse duration in ms (I-V curve).
        vc_post_pulse_duration: Post-pulse duration in ms (I-V curve).

    Returns:
        Protocol array in mV.

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    if protocol_type == "Step":
        protocol = patch_sim.step_voltage(
            duration=duration,
            voltage_amplitude=vc_voltage_amplitude,
            step_start=vc_step_start,
            step_duration=vc_step_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Ramp":
        protocol = patch_sim.ramp_voltage(
            duration=duration,
            start_voltage=vc_start_voltage,
            end_voltage=vc_end_voltage,
            ramp_start=vc_ramp_start,
            ramp_duration=vc_ramp_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "Pulse Train":
        protocol = patch_sim.pulse_train_voltage(
            duration=duration,
            pulse_amplitude=vc_pulse_amplitude,
            pulse_width=vc_pulse_width,
            pulse_interval=vc_pulse_interval,
            train_start=vc_train_start,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
    elif protocol_type == "I-V Curve":
        protocol = patch_sim.iv_curve_protocol(
            step_duration=duration,
            voltage_min=vc_voltage_min,
            voltage_max=vc_voltage_max,
            voltage_step=vc_voltage_step,
            pre_pulse_duration=vc_pre_pulse_duration,
            post_pulse_duration=vc_post_pulse_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
    else:
        raise ValueError(f"Unknown voltage protocol type: {protocol_type!r}")
    logger.debug(
        "build_voltage_protocol: type=%r duration=%.1f ms steps=%d",
        protocol_type,
        duration,
        len(protocol),
    )
    return protocol
