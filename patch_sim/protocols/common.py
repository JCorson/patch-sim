"""Shared helpers for protocol generation.

These private functions are used by both current and voltage clamp protocol
modules to avoid duplication.
"""

import numpy as np

# Default sampling frequency (Hz) for all protocol and simulation functions.
# Must equal :data:`~patch_sim.clamp_simulations.SIM_SAMPLING_FREQ` (40 kHz);
# the simulator reinterprets a protocol array as samples at its own rate, so a
# mismatched default silently stretches/compresses every protocol's duration
# when it reaches the simulator.  Prior to this change the default was 100 kHz
# while the simulator ran at 40 kHz, so a ``step_current(duration=900)`` call
# in a test produced a 90001-sample array that the simulator treated as
# 2250 ms instead of 900 ms — affecting depol-block timing on the rising LTS
# edge (#348), peak Ca²⁺ measurements during finite-duration stimuli, and
# passive-property fits whose stim windows are sized in absolute time.
# 40 kHz (dt = 25 µs) resolves Hodgkin-Huxley gating kinetics while keeping
# array sizes manageable.
DEFAULT_SAMPLING_FREQUENCY = 40_000.0


def _calculate_time_parameters(
    duration: float, sampling_frequency: float
) -> tuple[int, np.ndarray]:
    """Calculate common time parameters for protocol generation.

    Args:
        duration: Total duration in milliseconds
        sampling_frequency: Sampling frequency in Hz

    Returns:
        Tuple of (num_points, time_array)
    """
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1
    time_array = np.linspace(0, duration, num_points)
    return num_points, time_array


def _apply_time_window(
    time_array: np.ndarray,
    start_time: float,
    duration: float,
) -> np.ndarray:
    """Create a boolean mask for a time window.

    Args:
        time_array: Array of time points
        start_time: Start time of the window
        duration: Duration of the window

    Returns:
        Boolean mask array
    """
    end_time = start_time + duration
    return (time_array >= start_time) & (time_array <= end_time)


def _generate_step_protocol(
    duration: float,
    amplitude: float,
    baseline: float = 0.0,
    step_start: float = 0.0,
    step_duration: float | None = None,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a generic step protocol (current or voltage).

    This is a shared implementation for both current and voltage step protocols.

    Args:
        duration: Total duration of the protocol in milliseconds
        amplitude: Amplitude of the step
        baseline: Baseline value (0 for current, holding voltage for voltage)
        step_start: Time when the step begins in milliseconds
        step_duration: Duration of the step in milliseconds
        sampling_frequency: Sampling frequency in Hz

    Returns:
        Array of protocol values
    """
    if step_duration is None:
        step_duration = duration - step_start

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Initialize array with baseline
    protocol_array = np.full(num_points, baseline)

    # Set step values
    step_mask = _apply_time_window(time_array, step_start, step_duration)
    protocol_array[step_mask] = amplitude

    return protocol_array


def _generate_ramp_protocol(
    duration: float,
    start_value: float,
    end_value: float,
    baseline: float = 0.0,
    ramp_start: float = 0.0,
    ramp_duration: float | None = None,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a generic ramp protocol (current or voltage).

    This is a shared implementation for both current and voltage ramp protocols.

    Args:
        duration: Total duration of the protocol in milliseconds
        start_value: Starting value of the ramp
        end_value: Ending value of the ramp
        baseline: Baseline value (start_value for current, holding voltage for voltage)
        ramp_start: Time when the ramp begins in milliseconds
        ramp_duration: Duration of the ramp in milliseconds
        sampling_frequency: Sampling frequency in Hz

    Returns:
        Array of protocol values
    """
    if ramp_duration is None:
        ramp_duration = duration - ramp_start

    if ramp_duration == 0:
        raise ValueError("ramp_duration must not be zero.")

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Initialize array with baseline
    protocol_array = np.full(num_points, baseline)

    # Set ramp values
    ramp_mask = _apply_time_window(time_array, ramp_start, ramp_duration)

    if np.any(ramp_mask):
        ramp_times = time_array[ramp_mask]
        # Linear interpolation for the ramp
        protocol_array[ramp_mask] = start_value + (end_value - start_value) * (
            (ramp_times - ramp_start) / ramp_duration
        )

    return protocol_array


def _generate_pulse_train_protocol(
    duration: float,
    pulse_amplitude: float,
    pulse_width: float,
    pulse_interval: float,
    baseline: float = 0.0,
    train_start: float = 0.0,
    num_pulses: int | None = None,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a generic pulse train protocol (current or voltage).

    This is a shared implementation for both current and voltage pulse train protocols.

    Args:
        duration: Total duration of the protocol in milliseconds
        pulse_amplitude: Amplitude of each pulse
        pulse_width: Width of each pulse in milliseconds
        pulse_interval: Time between pulse onsets in milliseconds
        baseline: Baseline value (0 for current, holding voltage for voltage)
        train_start: Time when the pulse train begins in milliseconds
        num_pulses: Number of pulses in the train
        sampling_frequency: Sampling frequency in Hz

    Returns:
        Array of protocol values
    """
    if pulse_width >= pulse_interval:
        raise ValueError(
            "pulse_width must be less than pulse_interval to avoid overlapping pulses."
        )

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Initialize array with baseline
    protocol_array = np.full(num_points, baseline)

    # Calculate maximum number of pulses that fit
    if num_pulses is None:
        max_pulses = int((duration - train_start) / pulse_interval) + 1
    else:
        max_pulses = num_pulses

    # Generate each pulse
    for pulse_idx in range(max_pulses):
        pulse_start_time = train_start + pulse_idx * pulse_interval

        # Check if pulse fits within duration
        if pulse_start_time >= duration:
            break

        # Set pulse values
        pulse_mask = _apply_time_window(time_array, pulse_start_time, pulse_width)
        protocol_array[pulse_mask] = pulse_amplitude

    return protocol_array
