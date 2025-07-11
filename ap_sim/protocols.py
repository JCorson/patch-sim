"""
Current delivery protocols for use with current clamp experiments.

This module provides functions to generate typical current injection protocols
that can be used with the simulate_current_clamp function.
"""

import numpy as np
from typing import Optional


def step_current(
    duration: float,
    current_amplitude: float,
    step_start: float = 0.0,
    step_duration: Optional[float] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a step current protocol.

    Creates a current protocol with a rectangular step of specified amplitude
    and duration, useful for studying action potential threshold and firing patterns.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        current_amplitude (float): Amplitude of the current step in uA/cm^2.
        step_start (float): Time when the step begins in milliseconds. Default is 0.0.
        step_duration (float): Duration of the current step in milliseconds.
            If None, the step lasts for the entire duration.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    if step_duration is None:
        step_duration = duration - step_start

    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Create time array
    time_array = np.linspace(0, duration, num_points)

    # Initialize current array with zeros
    current_array = np.zeros(num_points)

    # Set step current values
    step_end = step_start + step_duration
    step_mask = (time_array >= step_start) & (time_array <= step_end)
    current_array[step_mask] = current_amplitude

    return current_array


def ramp_current(
    duration: float,
    start_current: float,
    end_current: float,
    ramp_start: float = 0.0,
    ramp_duration: Optional[float] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a ramp current protocol.

    Creates a current protocol with a linear ramp from start to end current,
    useful for studying firing frequency adaptation and rheobase determination.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        start_current (float): Starting current amplitude in uA/cm^2.
        end_current (float): Ending current amplitude in uA/cm^2.
        ramp_start (float): Time when the ramp begins in milliseconds. Default is 0.0.
        ramp_duration (float): Duration of the ramp in milliseconds.
            If None, the ramp lasts for the entire duration.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    if ramp_duration is None:
        ramp_duration = duration - ramp_start

    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Create time array
    time_array = np.linspace(0, duration, num_points)

    # Initialize current array with start current
    current_array = np.full(num_points, start_current)

    # Set ramp current values
    ramp_end = ramp_start + ramp_duration
    ramp_mask = (time_array >= ramp_start) & (time_array <= ramp_end)

    if np.any(ramp_mask):
        ramp_times = time_array[ramp_mask]
        # Linear interpolation for the ramp
        current_array[ramp_mask] = start_current + (end_current - start_current) * (
            (ramp_times - ramp_start) / ramp_duration
        )

    return current_array


def pulse_train(
    duration: float,
    pulse_amplitude: float,
    pulse_width: float,
    pulse_interval: float,
    train_start: float = 0.0,
    num_pulses: Optional[int] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a pulse train current protocol.

    Creates a series of rectangular current pulses with specified amplitude,
    width, and interval, useful for studying synaptic integration and
    temporal summation.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        pulse_amplitude (float): Amplitude of each pulse in uA/cm^2.
        pulse_width (float): Width of each pulse in milliseconds.
        pulse_interval (float): Time between pulse onsets in milliseconds.
        train_start (float): Time when the pulse train begins in milliseconds.
            Default is 0.0.
        num_pulses (int): Number of pulses in the train. If None, pulses
            continue until the end of the duration.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Create time array
    time_array = np.linspace(0, duration, num_points)

    # Initialize current array with zeros
    current_array = np.zeros(num_points)

    # Calculate maximum number of pulses that fit
    if num_pulses is None:
        max_pulses = int((duration - train_start) / pulse_interval) + 1
    else:
        max_pulses = num_pulses

    # Generate each pulse
    for pulse_idx in range(max_pulses):
        pulse_start_time = train_start + pulse_idx * pulse_interval
        pulse_end_time = pulse_start_time + pulse_width

        # Check if pulse fits within duration
        if pulse_start_time >= duration:
            break

        # Set pulse current values
        pulse_mask = (time_array >= pulse_start_time) & (time_array <= pulse_end_time)
        current_array[pulse_mask] = pulse_amplitude

    return current_array


def sinusoidal_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a sinusoidal current protocol.

    Creates a sinusoidal current waveform with specified DC offset, amplitude,
    and frequency, useful for studying frequency response and impedance.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        dc_offset (float): DC offset current in uA/cm^2.
        amplitude (float): Amplitude of the sinusoidal component in uA/cm^2.
        frequency (float): Frequency of the sinusoid in Hz.
        phase (float): Phase offset in radians. Default is 0.0.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Create time array in seconds for frequency calculation
    time_array = np.linspace(0, duration / 1000.0, num_points)

    # Generate sinusoidal current
    current_array = dc_offset + amplitude * np.sin(
        2 * np.pi * frequency * time_array + phase
    )

    return current_array


def chirp_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    start_frequency: float,
    end_frequency: float,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a chirp (frequency sweep) current protocol.

    Creates a sinusoidal current with linearly increasing frequency,
    useful for measuring frequency response characteristics.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        dc_offset (float): DC offset current in uA/cm^2.
        amplitude (float): Amplitude of the chirp in uA/cm^2.
        start_frequency (float): Starting frequency in Hz.
        end_frequency (float): Ending frequency in Hz.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Create time array in seconds
    time_array = np.linspace(0, duration / 1000.0, num_points)

    # Calculate instantaneous frequency
    freq_slope = (end_frequency - start_frequency) / (duration / 1000.0)

    # Generate chirp current using proper phase integration
    # Phase = 2π * ∫f(t)dt where f(t) = f0 + kt
    # ∫(f0 + kt)dt = f0*t + k*t²/2
    phase = (
        2 * np.pi * (start_frequency * time_array + 0.5 * freq_slope * time_array**2)
    )
    current_array = dc_offset + amplitude * np.sin(phase)

    return current_array


def noise_current(
    duration: float,
    mean_current: float,
    std_current: float,
    sampling_frequency: float = 100000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Gaussian white noise current protocol.

    Creates a current protocol with Gaussian-distributed random values,
    useful for studying stochastic resonance and noise effects.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        mean_current (float): Mean current value in uA/cm^2.
        std_current (float): Standard deviation of current in uA/cm^2.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.
        seed (int): Random seed for reproducibility. If None, uses random seed.

    Returns:
        np.ndarray: Array of current values in uA/cm^2.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Calculate time step and number of points
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1

    # Generate Gaussian noise
    current_array = np.random.normal(mean_current, std_current, num_points)

    return current_array
