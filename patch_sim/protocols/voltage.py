"""Voltage clamp protocol generation utilities.

This module provides functions to generate typical voltage stimulation
protocols that can be used with voltage clamp simulations.
"""

import numpy as np

from .common import (
    _apply_time_window,
    _calculate_time_parameters,
    _generate_pulse_train_protocol,
    _generate_ramp_protocol,
    _generate_step_protocol,
    DEFAULT_SAMPLING_FREQUENCY,
)


def step_voltage(
    duration: float,
    voltage_amplitude: float,
    step_start: float = 0.0,
    step_duration: float | None = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a step voltage protocol for voltage clamp experiments.

    Creates a voltage protocol with a rectangular step of specified amplitude
    and duration, useful for studying ionic currents and gating kinetics.

    Args:
        duration: Total duration of the protocol in milliseconds.
        voltage_amplitude: Amplitude of the voltage step in mV.
        step_start: Time when the step begins in milliseconds. Default is 0.0.
        step_duration: Duration of the voltage step in milliseconds.
            If None, the step lasts for the entire duration.
        holding_voltage: Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of voltage values in mV.
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
    ramp_duration: float | None = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a ramp voltage protocol for voltage clamp experiments.

    Creates a voltage protocol with a linear ramp from start to end voltage,
    useful for studying voltage-dependent activation and I-V relationships.

    Args:
        duration: Total duration of the protocol in milliseconds.
        start_voltage: Starting voltage in mV.
        end_voltage: Ending voltage in mV.
        ramp_start: Time when the ramp begins in milliseconds. Default is 0.0.
        ramp_duration: Duration of the ramp in milliseconds.
            If None, the ramp lasts for the entire duration.
        holding_voltage: Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of voltage values in mV.
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
    num_pulses: int | None = None,
    holding_voltage: float = -70.0,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate a pulse train voltage protocol for voltage clamp experiments.

    Creates a series of rectangular voltage pulses with specified amplitude,
    width, and interval, useful for studying synaptic currents and
    channel inactivation.

    Args:
        duration: Total duration of the protocol in milliseconds.
        pulse_amplitude: Amplitude of each pulse in mV.
        pulse_width: Width of each pulse in milliseconds.
        pulse_interval: Time between pulse onsets in milliseconds.
        train_start: Time when the pulse train begins in milliseconds.
            Default is 0.0.
        num_pulses: Number of pulses in the train. If None, pulses
            continue until the end of the duration.
        holding_voltage: Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of voltage values in mV.
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
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
) -> np.ndarray:
    """Generate an I-V curve protocol for voltage clamp experiments.

    Creates a series of voltage steps from minimum to maximum voltage,
    useful for characterizing current-voltage relationships.

    Args:
        step_duration: Duration of each voltage step in milliseconds.
        voltage_min: Minimum voltage in mV. Default is -100 mV.
        voltage_max: Maximum voltage in mV. Default is 60 mV.
        voltage_step: Voltage increment between steps in mV. Default is 10 mV.
        pre_pulse_duration: Duration before each step in milliseconds.
            Default is 10 ms.
        post_pulse_duration: Duration after each step in milliseconds.
            Default is 10 ms.
        holding_voltage: Holding voltage in mV. Default is -70.0 mV.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of voltage values in mV for the complete protocol.
    """
    # Calculate voltage steps
    n_steps = round((voltage_max - voltage_min) / voltage_step) + 1
    voltages = np.linspace(voltage_min, voltage_max, n_steps)

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
