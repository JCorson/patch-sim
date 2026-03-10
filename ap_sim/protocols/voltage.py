"""
Voltage clamp protocol generation utilities.

This module provides functions to generate typical voltage stimulation
protocols that can be used with voltage clamp simulations.
"""

from typing import Optional

import numpy as np

from .common import (
    _apply_time_window,
    _calculate_time_parameters,
    _generate_pulse_train_protocol,
    _generate_ramp_protocol,
    _generate_step_protocol,
)


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
