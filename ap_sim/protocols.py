"""
Protocol generation utilities for both current and voltage clamp experiments.

This module provides functions to generate typical stimulation protocols
that can be used with current and voltage clamp simulations.
"""

import numpy as np
from typing import Optional, Tuple


def _calculate_time_parameters(
    duration: float, sampling_frequency: float
) -> Tuple[int, np.ndarray]:
    """
    Calculate common time parameters for protocol generation.

    Args:
        duration: Total duration in milliseconds
        sampling_frequency: Sampling frequency in Hz

    Returns:
        Tuple of (num_points, time_array)
    """
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds
    num_points = int(duration / time_step) + 1
    time_array = np.linspace(0, duration, num_points)
    return num_points, time_array


def _apply_time_window(
    time_array: np.ndarray,
    start_time: float,
    duration: Optional[float] = None,
    end_time: Optional[float] = None,
) -> np.ndarray:
    """
    Create a boolean mask for a time window.

    Args:
        time_array: Array of time points
        start_time: Start time of the window
        duration: Duration of the window (if end_time not provided)
        end_time: End time of the window (if duration not provided)

    Returns:
        Boolean mask array
    """
    if end_time is None:
        if duration is None:
            raise ValueError("Either duration or end_time must be provided")
        end_time = start_time + duration

    return (time_array >= start_time) & (time_array <= end_time)


def _generate_step_protocol(
    duration: float,
    amplitude: float,
    baseline: float = 0.0,
    step_start: float = 0.0,
    step_duration: Optional[float] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a generic step protocol (current or voltage).

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
    ramp_duration: Optional[float] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a generic ramp protocol (current or voltage).

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
    num_pulses: Optional[int] = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a generic pulse train protocol (current or voltage).

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


# =============================================================================
# Current Clamp Protocols
# =============================================================================


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
    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Generate sinusoidal current
    time_array_seconds = time_array / 1000.0  # Convert to seconds
    current_array = dc_offset + amplitude * np.sin(
        2 * np.pi * frequency * time_array_seconds + phase
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
    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Calculate instantaneous frequency
    time_array_seconds = time_array / 1000.0  # Convert to seconds
    freq_slope = (end_frequency - start_frequency) / (duration / 1000.0)

    # Generate chirp current using proper phase integration
    # Phase = 2π * ∫f(t)dt where f(t) = f0 + kt
    # ∫(f0 + kt)dt = f0*t + k*t²/2
    phase = (
        2
        * np.pi
        * (
            start_frequency * time_array_seconds
            + 0.5 * freq_slope * time_array_seconds**2
        )
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
    # Create a local RNG instance to avoid mutating global state
    rng = np.random.default_rng(seed)

    # Calculate time step and number of points
    num_points, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Generate Gaussian noise
    current_array = rng.normal(mean_current, std_current, num_points)

    return current_array


# =============================================================================
# Voltage Clamp Protocols
# =============================================================================


def step_voltage(
    duration: float,
    voltage_amplitude: float,
    step_start: float = 0.0,
    step_duration: Optional[float] = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a step voltage protocol for voltage clamp experiments.

    Creates a voltage protocol with a rectangular step of specified amplitude
    and duration, useful for studying ionic currents and gating kinetics.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        voltage_amplitude (float): Amplitude of the voltage step in mV.
        step_start (float): Time when the step begins in milliseconds. Default is 0.0.
        step_duration (float): Duration of the voltage step in milliseconds.
            If None, the step lasts for the entire duration.
        holding_voltage (float): Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of voltage values in mV.
    """
    return _generate_step_protocol(
        duration,
        voltage_amplitude,
        baseline=holding_voltage,
        step_start=step_start,
        step_duration=step_duration,
        sampling_frequency=sampling_frequency,
    )


def ramp_voltage(
    duration: float,
    start_voltage: float,
    end_voltage: float,
    ramp_start: float = 0.0,
    ramp_duration: Optional[float] = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a ramp voltage protocol for voltage clamp experiments.

    Creates a voltage protocol with a linear ramp from start to end voltage,
    useful for studying voltage-dependent activation and I-V relationships.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        start_voltage (float): Starting voltage in mV.
        end_voltage (float): Ending voltage in mV.
        ramp_start (float): Time when the ramp begins in milliseconds. Default is 0.0.
        ramp_duration (float): Duration of the ramp in milliseconds.
            If None, the ramp lasts for the entire duration.
        holding_voltage (float): Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of voltage values in mV.
    """
    return _generate_ramp_protocol(
        duration,
        start_voltage,
        end_voltage,
        baseline=holding_voltage,
        ramp_start=ramp_start,
        ramp_duration=ramp_duration,
        sampling_frequency=sampling_frequency,
    )


def pulse_train_voltage(
    duration: float,
    pulse_amplitude: float,
    pulse_width: float,
    pulse_interval: float,
    train_start: float = 0.0,
    num_pulses: Optional[int] = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate a pulse train voltage protocol for voltage clamp experiments.

    Creates a series of rectangular voltage pulses with specified amplitude,
    width, and interval, useful for studying synaptic currents and
    channel inactivation.

    Parameters:
        duration (float): Total duration of the protocol in milliseconds.
        pulse_amplitude (float): Amplitude of each pulse in mV.
        pulse_width (float): Width of each pulse in milliseconds.
        pulse_interval (float): Time between pulse onsets in milliseconds.
        train_start (float): Time when the pulse train begins in milliseconds.
            Default is 0.0.
        num_pulses (int): Number of pulses in the train. If None, pulses
            continue until the end of the duration.
        holding_voltage (float): Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of voltage values in mV.
    """
    return _generate_pulse_train_protocol(
        duration,
        pulse_amplitude,
        pulse_width,
        pulse_interval,
        baseline=holding_voltage,
        train_start=train_start,
        num_pulses=num_pulses,
        sampling_frequency=sampling_frequency,
    )


def iv_curve_protocol(
    step_duration: float,
    voltage_min: float = -100.0,
    voltage_max: float = 60.0,
    voltage_step: float = 10.0,
    pre_pulse_duration: float = 10.0,
    post_pulse_duration: float = 10.0,
    holding_voltage: float = -70.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate an I-V curve protocol for voltage clamp experiments.

    Creates a series of voltage steps from minimum to maximum voltage,
    useful for characterizing current-voltage relationships.

    Parameters:
        step_duration (float): Duration of each voltage step in milliseconds.
        voltage_min (float): Minimum voltage in mV. Default is -100 mV.
        voltage_max (float): Maximum voltage in mV. Default is 60 mV.
        voltage_step (float): Voltage increment between steps in mV. Default is 10 mV.
        pre_pulse_duration (float): Duration before each step in milliseconds.
            Default is 10 ms.
        post_pulse_duration (float): Duration after each step in milliseconds.
            Default is 10 ms.
        holding_voltage (float): Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of voltage values in mV for the complete protocol.
    """
    # Calculate voltage steps
    voltages = np.arange(voltage_min, voltage_max + voltage_step, voltage_step)

    # Calculate total duration for one sweep
    sweep_duration = pre_pulse_duration + step_duration + post_pulse_duration
    total_duration = sweep_duration * len(voltages)

    num_points, time_array = _calculate_time_parameters(
        total_duration, sampling_frequency
    )

    # Initialize voltage array with holding voltage
    voltage_array = np.full(num_points, holding_voltage)

    # Generate each voltage step
    for i, voltage in enumerate(voltages):
        sweep_start = i * sweep_duration
        step_start = sweep_start + pre_pulse_duration

        step_mask = _apply_time_window(time_array, step_start, step_duration)
        voltage_array[step_mask] = voltage

    return voltage_array


def activation_protocol(
    test_duration: float,
    prepulse_voltage: float = -100.0,
    prepulse_duration: float = 500.0,
    test_voltage_min: float = -80.0,
    test_voltage_max: float = 60.0,
    voltage_step: float = 10.0,
    interpulse_duration: float = 10.0,
    holding_voltage: float = -70.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """
    Generate an activation protocol for voltage clamp experiments.

    Creates a protocol with prepulses followed by test pulses at different
    voltages, useful for studying channel activation kinetics.

    Parameters:
        test_duration (float): Duration of each test pulse in milliseconds.
        prepulse_voltage (float): Prepulse voltage in mV. Default is -100 mV.
        prepulse_duration (float): Duration of prepulse in milliseconds.
            Default is 500 ms.
        test_voltage_min (float): Minimum test voltage in mV. Default is -80 mV.
        test_voltage_max (float): Maximum test voltage in mV. Default is 60 mV.
        voltage_step (float): Voltage increment between test steps in mV.
            Default is 10 mV.
        interpulse_duration (float): Duration between prepulse and test pulse
            in milliseconds. Default is 10 ms.
        holding_voltage (float): Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency (float): Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        np.ndarray: Array of voltage values in mV for the complete protocol.
    """
    # Calculate test voltages
    test_voltages = np.arange(
        test_voltage_min, test_voltage_max + voltage_step, voltage_step
    )

    # Calculate total duration for one sweep
    sweep_duration = (
        prepulse_duration + interpulse_duration + test_duration + interpulse_duration
    )
    total_duration = sweep_duration * len(test_voltages)

    num_points, time_array = _calculate_time_parameters(
        total_duration, sampling_frequency
    )

    # Initialize voltage array with holding voltage
    voltage_array = np.full(num_points, holding_voltage)

    # Generate each sweep
    for i, test_voltage in enumerate(test_voltages):
        sweep_start = i * sweep_duration

        # Prepulse
        prepulse_start = sweep_start
        prepulse_mask = _apply_time_window(
            time_array, prepulse_start, prepulse_duration
        )
        voltage_array[prepulse_mask] = prepulse_voltage

        # Test pulse
        test_start = prepulse_start + prepulse_duration + interpulse_duration
        test_mask = _apply_time_window(time_array, test_start, test_duration)
        voltage_array[test_mask] = test_voltage

    return voltage_array
