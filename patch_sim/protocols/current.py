"""Current clamp protocol generation utilities.

This module provides functions to generate typical current stimulation
protocols that can be used with current clamp simulations.
"""

import numpy as np
from scipy.signal import chirp

from ..clamp_simulations import SIM_SAMPLING_FREQ
from .common import (
    _apply_time_window,
    _calculate_time_parameters,
    _generate_pulse_train_protocol,
    _generate_ramp_protocol,
    _generate_step_protocol,
)


def step_current(
    duration: float,
    current_amplitude: float,
    step_start: float = 0.0,
    step_duration: float | None = None,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
) -> np.ndarray:
    """Generate a step current protocol.

    Creates a current protocol with a rectangular step of specified amplitude
    and duration, useful for studying action potential threshold and firing patterns.

    Args:
        duration: Total duration of the protocol in milliseconds.
        current_amplitude: Amplitude of the current step in uA/cm^2.
        step_start: Time when the step begins in milliseconds. Default is 0.0.
        step_duration: Duration of the current step in milliseconds.
            If None, the step lasts for the entire duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_step_protocol(
        duration,
        current_amplitude,
        baseline=0.0,
        step_start=step_start,
        step_duration=step_duration,
        sampling_frequency=sampling_frequency,
    )


def ramp_current(
    duration: float,
    start_current: float,
    end_current: float,
    ramp_start: float = 0.0,
    ramp_duration: float | None = None,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
) -> np.ndarray:
    """Generate a ramp current protocol.

    Creates a current protocol with a linear ramp from start to end current,
    useful for studying firing frequency adaptation and rheobase determination.

    Args:
        duration: Total duration of the protocol in milliseconds.
        start_current: Starting current amplitude in uA/cm^2.
        end_current: Ending current amplitude in uA/cm^2.
        ramp_start: Time when the ramp begins in milliseconds. Default is 0.0.
        ramp_duration: Duration of the ramp in milliseconds.
            If None, the ramp lasts for the entire duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_ramp_protocol(
        duration,
        start_current,
        end_current,
        baseline=start_current,
        ramp_start=ramp_start,
        ramp_duration=ramp_duration,
        sampling_frequency=sampling_frequency,
    )


def pulse_train(
    duration: float,
    pulse_amplitude: float,
    pulse_width: float,
    pulse_interval: float,
    train_start: float = 0.0,
    num_pulses: int | None = None,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
) -> np.ndarray:
    """Generate a pulse train current protocol.

    Creates a series of rectangular current pulses with specified amplitude,
    width, and interval, useful for studying synaptic integration and
    temporal summation.

    Args:
        duration: Total duration of the protocol in milliseconds.
        pulse_amplitude: Amplitude of each pulse in uA/cm^2.
        pulse_width: Width of each pulse in milliseconds.
        pulse_interval: Time between pulse onsets in milliseconds.
        train_start: Time when the pulse train begins in milliseconds.
            Default is 0.0.
        num_pulses: Number of pulses in the train. If None, pulses
            continue until the end of the duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_pulse_train_protocol(
        duration,
        pulse_amplitude,
        pulse_width,
        pulse_interval,
        baseline=0.0,
        train_start=train_start,
        num_pulses=num_pulses,
        sampling_frequency=sampling_frequency,
    )


def sinusoidal_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    stimulus_start: float = 0.0,
    stimulus_duration: float = 0.0,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
) -> np.ndarray:
    """Generate a sinusoidal current protocol.

    Creates a sinusoidal current waveform with specified DC offset, amplitude,
    and frequency, useful for studying frequency response and impedance.

    The waveform starts at `stimulus_start` and lasts for `stimulus_duration`.
    The pre- and post-stimulus periods are filled with 0 current. When both
    `stimulus_start` and `stimulus_duration` are 0.0, the waveform fills the
    entire `duration`.

    Args:
        duration: Total duration of the protocol in milliseconds.
        dc_offset: DC offset current in uA/cm^2.
        amplitude: Amplitude of the sinusoidal component in uA/cm^2.
        frequency: Frequency of the sinusoid in Hz.
        phase: Phase offset in radians. Default is 0.0.
        stimulus_start: Time when the waveform begins in milliseconds.
            Default is 0.0.
        stimulus_duration: Duration of the sinusoidal stimulus in milliseconds.
            Default is 0.0, which fills the recording from stimulus_start to
            the end of duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.

    Returns:
        Array of current values in uA/cm^2.
    """
    if stimulus_duration == 0.0:
        stimulus_duration = duration - stimulus_start

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)
    current_array = np.zeros(num_points)

    stim_mask = _apply_time_window(time_array, stimulus_start, stimulus_duration)
    stim_times_seconds = (time_array[stim_mask] - stimulus_start) / 1000.0
    current_array[stim_mask] = dc_offset + amplitude * np.sin(
        2 * np.pi * frequency * stim_times_seconds + phase
    )

    return current_array


def chirp_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    start_frequency: float,
    end_frequency: float,
    stimulus_start: float = 0.0,
    stimulus_duration: float = 0.0,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
) -> np.ndarray:
    """Generate a chirp (frequency sweep) current protocol.

    Creates a sinusoidal current with linearly increasing frequency,
    useful for measuring frequency response characteristics.

    The waveform starts at `stimulus_start` and lasts for `stimulus_duration`.
    The pre- and post-stimulus periods are filled with 0 current. The frequency
    sweep spans `start_frequency` to `end_frequency` over `stimulus_duration`.
    When both `stimulus_start` and `stimulus_duration` are 0.0, the waveform
    fills the entire `duration`.

    Args:
        duration: Total duration of the protocol in milliseconds.
        dc_offset: DC offset current in uA/cm^2.
        amplitude: Amplitude of the chirp in uA/cm^2.
        start_frequency: Starting frequency in Hz.
        end_frequency: Ending frequency in Hz.
        stimulus_start: Time when the waveform begins in milliseconds.
            Default is 0.0.
        stimulus_duration: Duration of the chirp stimulus in milliseconds.
            Default is 0.0, which fills the recording from stimulus_start to
            the end of duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.

    Returns:
        Array of current values in uA/cm^2.
    """
    if stimulus_duration == 0.0:
        stimulus_duration = duration - stimulus_start

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)
    current_array = np.zeros(num_points)

    stim_mask = _apply_time_window(time_array, stimulus_start, stimulus_duration)
    stim_times_seconds = (time_array[stim_mask] - stimulus_start) / 1000.0
    stim_duration_seconds = stimulus_duration / 1000.0

    # phi=-90 converts scipy's cosine chirp to a sine chirp (sin = cos(x - 90°))
    current_array[stim_mask] = dc_offset + amplitude * chirp(
        stim_times_seconds,
        f0=start_frequency,
        t1=stim_duration_seconds,
        f1=end_frequency,
        method="linear",
        phi=-90,
    )

    return current_array


def noise_current(
    duration: float,
    mean_current: float,
    std_current: float,
    stimulus_start: float = 0.0,
    stimulus_duration: float = 0.0,
    sampling_frequency: float = SIM_SAMPLING_FREQ,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Gaussian white noise current protocol.

    Creates a current protocol with Gaussian-distributed random values,
    useful for studying stochastic resonance and noise effects.

    The noise starts at `stimulus_start` and lasts for `stimulus_duration`.
    The pre- and post-stimulus periods are filled with 0 current. When both
    `stimulus_start` and `stimulus_duration` are 0.0, the noise fills the
    entire `duration`.

    Args:
        duration: Total duration of the protocol in milliseconds.
        mean_current: Mean current value in uA/cm^2.
        std_current: Standard deviation of current in uA/cm^2.
        stimulus_start: Time when the noise begins in milliseconds.
            Default is 0.0.
        stimulus_duration: Duration of the noise stimulus in milliseconds.
            Default is 0.0, which fills the recording from stimulus_start to
            the end of duration.
        sampling_frequency: Sampling frequency in Hz; default SIM_SAMPLING_FREQ.
        seed: Random seed for reproducibility. If None, uses random seed.

    Returns:
        Array of current values in uA/cm^2.
    """
    if stimulus_duration == 0.0:
        stimulus_duration = duration - stimulus_start

    # Create a local RNG instance to avoid mutating global state
    rng = np.random.default_rng(seed)

    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)
    current_array = np.zeros(num_points)

    stim_mask = _apply_time_window(time_array, stimulus_start, stimulus_duration)
    n_stim_points = int(np.sum(stim_mask))
    current_array[stim_mask] = rng.normal(mean_current, std_current, n_stim_points)

    return current_array
