"""Profile simulate_voltage_clamp and simulate_current_clamp performance.

Generates a call-tree profile using cProfile for both simulation modes
on the default HodgkinHuxley model with a realistic 100 ms / 40 kHz protocol.

Usage::

    uv run --frozen scripts/profile_gating_state.py [output_file]

If *output_file* is given the sorted stats are also written there.
Defaults to stdout only.
"""

import cProfile
import io
import pstats
import sys
import time

import numpy as np

from ap_sim.clamp_simulations import (
    SIM_SAMPLING_FREQ,
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from ap_sim.hodgkin_huxley import HodgkinHuxley


def _build_voltage_protocol(duration_ms: float = 100.0) -> np.ndarray:
    """Build a hold→step→hold voltage-clamp protocol.

    Args:
        duration_ms: Total protocol duration in milliseconds.

    Returns:
        Voltage array in mV, length = duration_ms * SIM_SAMPLING_FREQ / 1000.
    """
    n = int(duration_ms * SIM_SAMPLING_FREQ / 1000.0)
    hold_v = -65.0
    step_v = 0.0
    protocol = np.full(n, hold_v)
    step_start = n // 4
    step_end = 3 * n // 4
    protocol[step_start:step_end] = step_v
    return protocol


def _build_current_protocol(duration_ms: float = 100.0) -> np.ndarray:
    """Build a step-pulse current-clamp protocol.

    Args:
        duration_ms: Total protocol duration in milliseconds.

    Returns:
        Current array in µA/cm², length = duration_ms * SIM_SAMPLING_FREQ / 1000.
    """
    n = int(duration_ms * SIM_SAMPLING_FREQ / 1000.0)
    protocol = np.zeros(n)
    step_start = n // 4
    step_end = 3 * n // 4
    protocol[step_start:step_end] = 10.0
    return protocol


def _profile_simulation(label: str, fn, *args) -> str:
    """Run *fn* under cProfile and return a formatted stats string.

    Args:
        label: Human-readable label for the simulation (used in output header).
        fn: Simulation function to profile.
        *args: Arguments forwarded to *fn*.

    Returns:
        Formatted stats string sorted by cumulative time (top 30 functions).
    """
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    fn(*args)
    pr.disable()
    elapsed = time.perf_counter() - t0

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(30)

    header = f"\n{'=' * 70}\n{label}  (wall time: {elapsed:.3f} s)\n{'=' * 70}\n"
    return header + buf.getvalue()


def main() -> None:
    """Run the profiler and print/save results.

    Reads an optional output file path from sys.argv[1].
    """
    output_file: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    neuron = HodgkinHuxley()
    v_protocol = _build_voltage_protocol()
    i_protocol = _build_current_protocol()

    vc_stats = _profile_simulation(
        "simulate_voltage_clamp", simulate_voltage_clamp, neuron, v_protocol
    )
    cc_stats = _profile_simulation(
        "simulate_current_clamp", simulate_current_clamp, neuron, i_protocol
    )

    report = vc_stats + "\n" + cc_stats

    print(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\nProfile written to: {output_file}")


if __name__ == "__main__":
    main()
