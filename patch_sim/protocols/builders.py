"""Pure functions for building simulation protocol arrays.

These functions take explicit parameters and return a list of
(stimulus_array, sweep_label) pairs.
"""

import logging

import numpy as np

from .current import (
    chirp_current,
    noise_current,
    pulse_train,
    ramp_current,
    sinusoidal_current,
    step_current,
)
from .voltage import (
    pulse_train_voltage,
    ramp_voltage,
    step_voltage,
)

logger = logging.getLogger(__name__)


def build_current_protocol(
    protocol_type: str,
    sampling_frequency: float,
    pre_stimulus_duration: float = 10.0,
    stimulus_duration: float = 30.0,
    post_stimulus_duration: float = 10.0,
    min_stimulus: float = 10.0,
    max_stimulus: float = 10.0,
    stimulus_step: float = 0.0,
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
        min_stimulus: Step current amplitude in µA/cm² for a single-step, or
            minimum current when running a multi-sweep range.
        max_stimulus: Maximum current for multi-sweep range in µA/cm².  Must
            be >= min_stimulus.
        stimulus_step: Step size for multi-sweep range in µA/cm².  Use 0.0
            (or set min_stimulus == max_stimulus) for a single-step protocol.
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
        List of (stimulus_array, sweep_label) pairs.  Single-step protocols
        return a one-element list with an empty label.  Multi-sweep Step
        protocols return one entry per current level labelled
        "+X.X µA/cm²" / "-X.X µA/cm²".

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    total_duration = pre_stimulus_duration + stimulus_duration + post_stimulus_duration
    result: list[tuple[np.ndarray, str]]
    if protocol_type == "Step":
        if stimulus_step < 0.0:
            raise ValueError("stimulus_step must be >= 0.0")
        if min_stimulus > max_stimulus:
            raise ValueError(
                f"min_stimulus ({min_stimulus}) must be"
                f" <= max_stimulus ({max_stimulus})"
            )
        if min_stimulus != max_stimulus and stimulus_step == 0.0:
            raise ValueError(
                "stimulus_step must be > 0.0 when min_stimulus != max_stimulus"
            )
        if min_stimulus == max_stimulus:
            protocol = step_current(
                duration=total_duration,
                current_amplitude=min_stimulus,
                step_start=pre_stimulus_duration,
                step_duration=stimulus_duration,
                sampling_frequency=sampling_frequency,
            )
            result = [(protocol, "")]
        else:
            n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
            currents = np.linspace(min_stimulus, max_stimulus, n_steps)
            result = [
                (
                    step_current(
                        duration=total_duration,
                        current_amplitude=float(current),
                        step_start=pre_stimulus_duration,
                        step_duration=stimulus_duration,
                        sampling_frequency=sampling_frequency,
                    ),
                    f"{current:+.1f} µA/cm²",
                )
                for current in currents
            ]
    elif protocol_type == "Ramp":
        result = [
            (
                ramp_current(
                    duration=total_duration,
                    start_current=start_current,
                    end_current=end_current,
                    ramp_start=pre_stimulus_duration,
                    ramp_duration=stimulus_duration,
                    sampling_frequency=sampling_frequency,
                ),
                "",
            )
        ]
    elif protocol_type == "Pulse Train":
        result = [
            (
                pulse_train(
                    duration=total_duration,
                    pulse_amplitude=pulse_amplitude,
                    pulse_width=pulse_width,
                    pulse_interval=pulse_interval,
                    train_start=pre_stimulus_duration,
                    sampling_frequency=sampling_frequency,
                ),
                "",
            )
        ]
    elif protocol_type == "Sinusoidal":
        result = [
            (
                sinusoidal_current(
                    duration=total_duration,
                    dc_offset=dc_offset,
                    amplitude=amplitude,
                    frequency=frequency,
                    stimulus_start=pre_stimulus_duration,
                    stimulus_duration=stimulus_duration,
                    sampling_frequency=sampling_frequency,
                ),
                "",
            )
        ]
    elif protocol_type == "Chirp":
        result = [
            (
                chirp_current(
                    duration=total_duration,
                    dc_offset=dc_offset,
                    amplitude=amplitude,
                    start_frequency=start_frequency,
                    end_frequency=end_frequency,
                    stimulus_start=pre_stimulus_duration,
                    stimulus_duration=stimulus_duration,
                    sampling_frequency=sampling_frequency,
                ),
                "",
            )
        ]
    elif protocol_type == "Noise":
        result = [
            (
                noise_current(
                    duration=total_duration,
                    mean_current=mean_current,
                    std_current=std_current,
                    stimulus_start=pre_stimulus_duration,
                    stimulus_duration=stimulus_duration,
                    sampling_frequency=sampling_frequency,
                ),
                "",
            )
        ]
    else:
        raise ValueError(f"Unknown current protocol type: {protocol_type!r}")
    logger.debug(
        "build_current_protocol: type=%r total_duration=%.1f ms sweeps=%d",
        protocol_type,
        total_duration,
        len(result),
    )
    return result


def build_voltage_protocol(
    protocol_type: str,
    sampling_frequency: float,
    pre_stimulus_duration: float = 10.0,
    stimulus_duration: float = 30.0,
    post_stimulus_duration: float = 10.0,
    vc_holding_voltage: float = -70.0,
    start_voltage: float = -70.0,
    end_voltage: float = 40.0,
    pulse_amplitude: float = 20.0,
    pulse_width: float = 2.0,
    pulse_interval: float = 10.0,
    min_stimulus: float = 0.0,
    max_stimulus: float = 0.0,
    stimulus_step: float = 0.0,
) -> list[tuple[np.ndarray, str]]:
    """Build voltage clamp protocol arrays from explicit parameters.

    Args:
        protocol_type: One of "Step", "Ramp", or "Pulse Train".
        sampling_frequency: Sampling frequency in Hz.
        pre_stimulus_duration: Duration before the stimulus in ms.
        stimulus_duration: Duration of the stimulus in ms.
        post_stimulus_duration: Duration after the stimulus in ms.
        vc_holding_voltage: Holding voltage in mV.
        start_voltage: Ramp start voltage in mV.
        end_voltage: Ramp end voltage in mV.
        pulse_amplitude: Pulse amplitude in mV.
        pulse_width: Pulse width in ms.
        pulse_interval: Interval between pulse starts in ms.
        min_stimulus: Step voltage amplitude in mV for a single-step, or
            minimum voltage when running a multi-sweep range.
        max_stimulus: Maximum voltage for multi-sweep range in mV.  Must
            be >= min_stimulus.
        stimulus_step: Step size for multi-sweep range in mV.  Use 0.0
            (or set min_stimulus == max_stimulus) for a single-step protocol.

    Returns:
        List of (stimulus_array, sweep_label) pairs.  Single-step protocols
        return a one-element list with an empty label.  Multi-sweep Step
        protocols return one entry per voltage level labelled
        "+XX mV" / "-XX mV".

    Raises:
        ValueError: If protocol_type is unrecognized or parameters are invalid.
    """
    total_duration = pre_stimulus_duration + stimulus_duration + post_stimulus_duration
    result: list[tuple[np.ndarray, str]]
    if protocol_type == "Step":
        if stimulus_step < 0.0:
            raise ValueError("stimulus_step must be >= 0.0")
        if min_stimulus > max_stimulus:
            raise ValueError(
                f"min_stimulus ({min_stimulus}) must be"
                f" <= max_stimulus ({max_stimulus})"
            )
        if min_stimulus != max_stimulus and stimulus_step == 0.0:
            raise ValueError(
                "stimulus_step must be > 0.0 when min_stimulus != max_stimulus"
            )
        is_single_step = min_stimulus == max_stimulus
        if is_single_step:
            protocol = step_voltage(
                duration=total_duration,
                voltage_amplitude=min_stimulus,
                step_start=pre_stimulus_duration,
                step_duration=stimulus_duration,
                holding_voltage=vc_holding_voltage,
                sampling_frequency=sampling_frequency,
            )
            result = [(protocol, "")]
        else:
            n_steps = round((max_stimulus - min_stimulus) / stimulus_step) + 1
            voltages = np.linspace(min_stimulus, max_stimulus, n_steps)
            result = [
                (
                    step_voltage(
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
    elif protocol_type == "Ramp":
        protocol = ramp_voltage(
            duration=total_duration,
            start_voltage=start_voltage,
            end_voltage=end_voltage,
            ramp_start=pre_stimulus_duration,
            ramp_duration=stimulus_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
        result = [(protocol, "")]
    elif protocol_type == "Pulse Train":
        protocol = pulse_train_voltage(
            duration=total_duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=pre_stimulus_duration,
            holding_voltage=vc_holding_voltage,
            sampling_frequency=sampling_frequency,
        )
        result = [(protocol, "")]
    else:
        raise ValueError(f"Unknown voltage protocol type: {protocol_type!r}")
    logger.debug(
        "build_voltage_protocol: type=%r total_duration=%.1f ms sweeps=%d",
        protocol_type,
        total_duration,
        len(result),
    )
    return result
