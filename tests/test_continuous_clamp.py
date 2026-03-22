"""Tests for simulate_current_clamp_from_state and simulate_voltage_clamp_from_state.

These functions power the continuous simulation loop in the UI, carrying terminal
neuron state across loop iterations to provide true physiological continuity.
"""

import numpy as np
import pytest

from patch_sim.clamp_simulations import (
    SIM_SAMPLING_FREQ,
    _initialize_gating_variables,
    simulate_current_clamp,
    simulate_current_clamp_from_state,
    simulate_voltage_clamp,
    simulate_voltage_clamp_from_state,
)


def _make_steps(duration_ms: float, current: float = 0.0) -> np.ndarray:
    """Return a constant-current array for the given duration.

    Args:
        duration_ms: Duration of the stimulus in milliseconds.
        current: Constant current value in µA/cm².

    Returns:
        NumPy array of current values at the simulation sampling rate.
    """
    num_steps = int(duration_ms / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    return np.full(num_steps, current)


def _make_voltage_steps(duration_ms: float, voltage: float = -70.0) -> np.ndarray:
    """Return a constant-voltage array for the given duration.

    Args:
        duration_ms: Duration of the command in milliseconds.
        voltage: Constant voltage value in mV.

    Returns:
        NumPy array of voltage values at the simulation sampling rate.
    """
    num_steps = int(duration_ms / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    return np.full(num_steps, voltage)


# ---------------------------------------------------------------------------
# Current clamp _from_state tests
# ---------------------------------------------------------------------------


def test_cc_from_state_voltage_continuity(hh_model):
    """Terminal V of run 1 equals initial V at t=0 of run 2.

    Splitting a simulation into two halves and passing terminal state as
    initial conditions must produce seamless continuity at the junction.
    """
    stimulus = _make_steps(10.0, current=20.0)

    df1 = simulate_current_clamp(hh_model, stimulus)

    last_V = float(df1["voltage"].iloc[-1])
    gating_cols = [
        col
        for col in df1.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_gating = {col: float(df1[col].iloc[-1]) for col in gating_cols}

    df2 = simulate_current_clamp_from_state(
        hh_model,
        stimulus,
        initial_V=last_V,
        initial_gating_state=last_gating,
    )

    assert df2["voltage"].iloc[0] == pytest.approx(last_V)


def test_cc_from_state_gating_continuity(hh_model):
    """Terminal gating state of run 1 equals initial gating state at t=0 of run 2.

    Verifies that all classic gating variables (n, m, h) are faithfully
    carried forward.
    """
    stimulus = _make_steps(10.0, current=20.0)

    df1 = simulate_current_clamp(hh_model, stimulus)

    # Map HH variable names as column names
    gating_cols = [
        col
        for col in df1.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_gating = {col: float(df1[col].iloc[-1]) for col in gating_cols}

    df2 = simulate_current_clamp_from_state(
        hh_model,
        stimulus,
        initial_V=float(df1["voltage"].iloc[-1]),
        initial_gating_state=last_gating,
    )

    for col, val in last_gating.items():
        assert df2[col].iloc[0] == pytest.approx(val, abs=1e-9)


def test_cc_from_state_split_matches_full_run(hh_model):
    """Splitting a run into two halves via from_state reproduces a single full run.

    Concatenating a first-half run (fresh initial conditions) with a second-half
    run (carrying terminal state) must match a single full run over the same
    total duration.
    """
    half = _make_steps(5.0, current=0.0)  # zero current so state is stable
    full = _make_steps(10.0, current=0.0)

    df_full = simulate_current_clamp(hh_model, full)

    df1 = simulate_current_clamp(hh_model, half)
    gating_cols = [
        col
        for col in df1.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_gating = {col: float(df1[col].iloc[-1]) for col in gating_cols}

    df2 = simulate_current_clamp_from_state(
        hh_model,
        half,
        initial_V=float(df1["voltage"].iloc[-1]),
        initial_gating_state=last_gating,
    )

    # The terminal values of the split run and the full run must match.
    assert df2["voltage"].iloc[-1] == pytest.approx(
        df_full["voltage"].iloc[-1], abs=1e-6
    )
    for col in gating_cols:
        assert df2[col].iloc[-1] == pytest.approx(df_full[col].iloc[-1], abs=1e-6)


def test_cc_from_state_empty_raises(hh_model):
    """simulate_current_clamp_from_state raises ValueError for empty stimulus.

    Ensures that the same input validation as simulate_current_clamp is
    preserved in the _from_state variant.
    """
    gating = _initialize_gating_variables(hh_model, hh_model.v_rest)
    with pytest.raises(ValueError, match="must not be empty"):
        simulate_current_clamp_from_state(
            hh_model,
            np.array([]),
            initial_V=hh_model.v_rest,
            initial_gating_state=gating,
        )


# ---------------------------------------------------------------------------
# Voltage clamp _from_state tests
# ---------------------------------------------------------------------------


def test_vc_from_state_gating_continuity(hh_model):
    """Terminal gating state of run 1 equals initial gating state at t=0 of run 2.

    Verifies that voltage clamp gating state is faithfully carried forward.
    """
    protocol = _make_voltage_steps(10.0, voltage=0.0)  # depolarised hold

    df1 = simulate_voltage_clamp(hh_model, protocol)

    gating_cols = [
        col
        for col in df1.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_gating = {col: float(df1[col].iloc[-1]) for col in gating_cols}

    df2 = simulate_voltage_clamp_from_state(
        hh_model,
        protocol,
        initial_gating_state=last_gating,
    )

    for col, val in last_gating.items():
        assert df2[col].iloc[0] == pytest.approx(val, abs=1e-9)


def test_vc_from_state_split_matches_full_run(hh_model):
    """Splitting a voltage clamp run into two halves via from_state matches a full run.

    Concatenating first-half and second-half runs via from_state must reproduce
    the terminal state of a single full-duration run.
    """
    hold = -70.0  # holding at rest — stable, easy to reproduce
    half = _make_voltage_steps(5.0, voltage=hold)
    full = _make_voltage_steps(10.0, voltage=hold)

    df_full = simulate_voltage_clamp(hh_model, full)

    df1 = simulate_voltage_clamp(hh_model, half)
    gating_cols = [
        col
        for col in df1.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_gating = {col: float(df1[col].iloc[-1]) for col in gating_cols}

    df2 = simulate_voltage_clamp_from_state(
        hh_model,
        half,
        initial_gating_state=last_gating,
    )

    for col in gating_cols:
        assert df2[col].iloc[-1] == pytest.approx(df_full[col].iloc[-1], abs=1e-6)


def test_vc_from_state_empty_raises(hh_model):
    """simulate_voltage_clamp_from_state raises ValueError for empty protocol.

    Ensures that the same input validation as simulate_voltage_clamp is
    preserved in the _from_state variant.
    """
    gating = _initialize_gating_variables(hh_model, hh_model.v_rest)
    with pytest.raises(ValueError, match="must not be empty"):
        simulate_voltage_clamp_from_state(
            hh_model,
            np.array([]),
            initial_gating_state=gating,
        )


# ---------------------------------------------------------------------------
# Multi-iteration sequential _from_state tests (continuous loop simulation)
# ---------------------------------------------------------------------------


def _extract_terminal_cc_state(df, hh_model):
    """Return (last_V, last_gating_dict) from a current-clamp result DataFrame."""
    gating_cols = [
        col
        for col in df.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    last_V = float(df["voltage"].iloc[-1])
    last_gating = {col: float(df[col].iloc[-1]) for col in gating_cols}
    return last_V, last_gating


def _extract_terminal_vc_state(df, hh_model):
    """Return last_gating_dict from a voltage-clamp result DataFrame."""
    gating_cols = [
        col
        for col in df.columns
        if col in {gv.name for gv in hh_model.all_gating_variables}
    ]
    return {col: float(df[col].iloc[-1]) for col in gating_cols}


def test_cc_three_iteration_loop_state_continuity(hh_model):
    """Three chained CC _from_state calls maintain junction continuity.

    Simulates three consecutive iterations of the continuous loop.  At each
    junction the terminal voltage and gating state of iteration N must equal
    the initial voltage and gating state at t=0 of iteration N+1.
    """
    stimulus = _make_steps(5.0, current=15.0)

    # Iteration 1 — fresh start
    df1 = simulate_current_clamp(hh_model, stimulus)
    last_V1, last_gating1 = _extract_terminal_cc_state(df1, hh_model)

    # Iteration 2 — carry forward from iteration 1
    df2 = simulate_current_clamp_from_state(
        hh_model, stimulus, initial_V=last_V1, initial_gating_state=last_gating1
    )
    assert df2["voltage"].iloc[0] == pytest.approx(last_V1)
    for col, val in last_gating1.items():
        assert df2[col].iloc[0] == pytest.approx(val, abs=1e-9)

    last_V2, last_gating2 = _extract_terminal_cc_state(df2, hh_model)

    # Iteration 3 — carry forward from iteration 2
    df3 = simulate_current_clamp_from_state(
        hh_model, stimulus, initial_V=last_V2, initial_gating_state=last_gating2
    )
    assert df3["voltage"].iloc[0] == pytest.approx(last_V2)
    for col, val in last_gating2.items():
        assert df3[col].iloc[0] == pytest.approx(val, abs=1e-9)


def test_cc_three_iteration_loop_matches_full_run(hh_model):
    """Three chained CC iterations match a single equivalent full run.

    Running three back-to-back _from_state iterations over a zero-current
    stimulus must produce the same terminal state as a single run of three
    times the duration (zero current keeps the membrane at rest throughout).
    """
    segment = _make_steps(5.0, current=0.0)
    full = _make_steps(15.0, current=0.0)

    df_full = simulate_current_clamp(hh_model, full)

    df1 = simulate_current_clamp(hh_model, segment)
    last_V1, last_gating1 = _extract_terminal_cc_state(df1, hh_model)

    df2 = simulate_current_clamp_from_state(
        hh_model, segment, initial_V=last_V1, initial_gating_state=last_gating1
    )
    last_V2, last_gating2 = _extract_terminal_cc_state(df2, hh_model)

    df3 = simulate_current_clamp_from_state(
        hh_model, segment, initial_V=last_V2, initial_gating_state=last_gating2
    )
    last_V3, last_gating3 = _extract_terminal_cc_state(df3, hh_model)

    gating_cols = list(last_gating3.keys())
    assert last_V3 == pytest.approx(float(df_full["voltage"].iloc[-1]), abs=1e-6)
    for col in gating_cols:
        assert last_gating3[col] == pytest.approx(
            float(df_full[col].iloc[-1]), abs=1e-6
        )


def test_vc_three_iteration_loop_state_continuity(hh_model):
    """Three chained VC _from_state calls maintain junction continuity.

    Simulates three consecutive voltage-clamp iterations.  At each junction
    the terminal gating state of iteration N must equal the initial gating
    state at t=0 of iteration N+1.
    """
    protocol = _make_voltage_steps(5.0, voltage=-20.0)

    # Iteration 1 — fresh start
    df1 = simulate_voltage_clamp(hh_model, protocol)
    last_gating1 = _extract_terminal_vc_state(df1, hh_model)

    # Iteration 2
    df2 = simulate_voltage_clamp_from_state(
        hh_model, protocol, initial_gating_state=last_gating1
    )
    for col, val in last_gating1.items():
        assert df2[col].iloc[0] == pytest.approx(val, abs=1e-9)

    last_gating2 = _extract_terminal_vc_state(df2, hh_model)

    # Iteration 3
    df3 = simulate_voltage_clamp_from_state(
        hh_model, protocol, initial_gating_state=last_gating2
    )
    for col, val in last_gating2.items():
        assert df3[col].iloc[0] == pytest.approx(val, abs=1e-9)


def test_vc_three_iteration_loop_matches_full_run(hh_model):
    """Three chained VC iterations match a single equivalent full run.

    Running three back-to-back _from_state iterations at the holding voltage
    must produce the same terminal gating state as a single run of three times
    the duration.
    """
    hold = -70.0
    segment = _make_voltage_steps(5.0, voltage=hold)
    full = _make_voltage_steps(15.0, voltage=hold)

    df_full = simulate_voltage_clamp(hh_model, full)

    df1 = simulate_voltage_clamp(hh_model, segment)
    last_gating1 = _extract_terminal_vc_state(df1, hh_model)

    df2 = simulate_voltage_clamp_from_state(
        hh_model, segment, initial_gating_state=last_gating1
    )
    last_gating2 = _extract_terminal_vc_state(df2, hh_model)

    df3 = simulate_voltage_clamp_from_state(
        hh_model, segment, initial_gating_state=last_gating2
    )
    last_gating3 = _extract_terminal_vc_state(df3, hh_model)

    for col in last_gating3:
        assert last_gating3[col] == pytest.approx(
            float(df_full[col].iloc[-1]), abs=1e-6
        )
