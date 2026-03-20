"""Profile simulate_voltage_clamp and simulate_current_clamp performance.

Generates a call-tree profile using cProfile for both simulation modes
on the default HodgkinHuxley model with a realistic 100 ms / 40 kHz protocol.

When Numba is installed, a warm-up call is made before profiling so that JIT
compilation time is excluded from the measurement.

Usage::

    uv run --frozen scripts/profile_gating_state.py [output_file]

If *output_file* is given the sorted stats are also written there.
Defaults to stdout only.

A comparison summary is always printed at the end showing wall times across
the three implementation strategies:

- **dict-based** — original gating state as ``dict[str, float]``
- **array-based** — gating state as a flat numpy array (Step 2)
- **Numba JIT** — JIT-compiled inner loops via ``numba_kernel`` (Step 3)
"""

import cProfile
import io
import pstats
import sys
import time

import numpy as np

from ap_sim.clamp_simulations import (
    HAS_NUMBA,
    SIM_SAMPLING_FREQ,
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from ap_sim.hodgkin_huxley import HodgkinHuxley

# Reference wall times recorded on this machine for earlier implementations.
# Dict-based: original implementation before Issue #65 refactoring.
# Array-based: Step 1+2 of Issue #65 (numpy array gating state, no Numba).
_DICT_BASED_VC_S = 0.339
_DICT_BASED_CC_S = 0.668
_ARRAY_BASED_VC_S = 0.450
_ARRAY_BASED_CC_S = 1.072


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


def _profile_simulation(label: str, fn, *args) -> tuple[str, float]:
    """Run *fn* under cProfile and return a formatted stats string and wall time.

    Args:
        label: Human-readable label for the simulation (used in output header).
        fn: Simulation function to profile.
        *args: Arguments forwarded to *fn*.

    Returns:
        Tuple of (formatted_stats_string, wall_time_seconds) where the stats
        string is sorted by cumulative time (top 30 functions).
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
    return header + buf.getvalue(), elapsed


def _warmup(
    neuron: HodgkinHuxley, v_protocol: np.ndarray, i_protocol: np.ndarray
) -> None:
    """Run short warm-up simulations to trigger Numba JIT compilation.

    The first call to a Numba-compiled function incurs compilation overhead.
    This warm-up uses a tiny 10-step protocol so compilation completes before
    the real profiling run.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        v_protocol: Full voltage protocol (only first 10 steps are used).
        i_protocol: Full current protocol (only first 10 steps are used).
    """
    print("Numba detected — running warm-up to trigger JIT compilation...")
    t0 = time.perf_counter()
    simulate_voltage_clamp(neuron, v_protocol[:10])
    simulate_current_clamp(neuron, i_protocol[:10])
    elapsed = time.perf_counter() - t0
    print(f"  Warm-up complete ({elapsed:.1f} s including compilation)\n")


def _comparison_summary(vc_s: float, cc_s: float) -> str:
    """Format a comparison table of wall times across all three strategies.

    Args:
        vc_s: Current-run wall time for voltage-clamp in seconds.
        cc_s: Current-run wall time for current-clamp in seconds.

    Returns:
        Formatted multi-line summary string.
    """
    label = "Numba JIT" if HAS_NUMBA else "array-based (no Numba)"
    rows = [
        ("dict-based (ref)", _DICT_BASED_VC_S, _DICT_BASED_CC_S),
        ("array-based (ref)", _ARRAY_BASED_VC_S, _ARRAY_BASED_CC_S),
        (label, vc_s, cc_s),
    ]
    lines = [
        "\n" + "=" * 70,
        "  Comparison summary (100 ms / 40 kHz, post-JIT-warmup)",
        "=" * 70,
        f"  {'Strategy':<22} {'VC (s)':>10} {'CC (s)':>10}",
        "  " + "-" * 44,
    ]
    for name, vc, cc in rows:
        lines.append(f"  {name:<22} {vc:>10.3f} {cc:>10.3f}")
    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


def main() -> None:
    """Run the profiler and print/save results.

    Reads an optional output file path from sys.argv[1].
    """
    output_file: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    neuron = HodgkinHuxley()
    v_protocol = _build_voltage_protocol()
    i_protocol = _build_current_protocol()

    if HAS_NUMBA:
        _warmup(neuron, v_protocol, i_protocol)

    vc_stats, vc_s = _profile_simulation(
        "simulate_voltage_clamp", simulate_voltage_clamp, neuron, v_protocol
    )
    cc_stats, cc_s = _profile_simulation(
        "simulate_current_clamp", simulate_current_clamp, neuron, i_protocol
    )

    summary = _comparison_summary(vc_s, cc_s)
    report = vc_stats + "\n" + cc_stats + "\n" + summary

    print(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Profile written to: {output_file}")


if __name__ == "__main__":
    main()
