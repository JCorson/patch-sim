"""Profile build_figure() for the Na+ Channel Activation full dataset.

Run with:
    uv run --frozen python scripts/profile_plot.py
"""

import cProfile
import io
import pathlib
import sys

# Ensure the project root is on sys.path so that patch_sim and patch_sim_ui
# are importable when the script is run from any directory.
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pstats  # noqa: E402

import patch_sim  # noqa: E402
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ  # noqa: E402
from patch_sim_ui.constants import SWEEP_COLORS, VOLTAGE_CLAMP  # noqa: E402
from patch_sim_ui.plotting import Sweep, TraceVisibility, build_figure  # noqa: E402


# ---------------------------------------------------------------------------
# Preset parameters: Na+ Channel Activation
# ---------------------------------------------------------------------------
PRESET = dict(
    g_Na=120.0,
    g_K=0.0,  # K+ blocked to isolate Na+
    g_L=0.3,
    C_m=1.0,
    v_rest=-65.0,
    Na_out=145.0,
    Na_in=15.0,
    K_out=5.0,
    K_in=140.0,
    Cl_out=120.0,
    Cl_in=10.0,
    Ca_out=2.0,
    Ca_in=0.0001,
    T=310.15,
    # Protocol
    duration=20.0,
    vc_holding_voltage=-70.0,
    vc_baseline_duration=50.0,
    vc_test_voltage_min=-60.0,
    vc_test_voltage_max=60.0,
    vc_voltage_step=10.0,
    vc_interpulse_duration=5.0,
)


def build_sweeps() -> list[Sweep]:
    """Simulate all Na+ activation sweeps and return Sweep objects.

    Returns:
        List of Sweep objects, one per test voltage.
    """
    neuron = patch_sim.HodgkinHuxley(
        g_Na=PRESET["g_Na"],
        g_K=PRESET["g_K"],
        g_L=PRESET["g_L"],
        C_m=PRESET["C_m"],
        v_rest=PRESET["v_rest"],
        Na_out=PRESET["Na_out"],
        Na_in=PRESET["Na_in"],
        K_out=PRESET["K_out"],
        K_in=PRESET["K_in"],
        Cl_out=PRESET["Cl_out"],
        Cl_in=PRESET["Cl_in"],
        Ca_out=PRESET["Ca_out"],
        Ca_in=PRESET["Ca_in"],
        T=PRESET["T"],
    )

    voltage_range = PRESET["vc_test_voltage_max"] - PRESET["vc_test_voltage_min"]
    n_steps = round(voltage_range / PRESET["vc_voltage_step"]) + 1
    test_voltages = np.linspace(
        PRESET["vc_test_voltage_min"], PRESET["vc_test_voltage_max"], n_steps
    )

    protocols = [
        patch_sim.activation_sweep(
            test_duration=PRESET["duration"],
            test_voltage=float(v),
            baseline_duration=PRESET["vc_baseline_duration"],
            interpulse_duration=PRESET["vc_interpulse_duration"],
            holding_voltage=PRESET["vc_holding_voltage"],
            sampling_frequency=SIM_SAMPLING_FREQ,
        )
        for v in test_voltages
    ]

    sweeps: list[Sweep] = []
    for df, voltage, protocol in zip(
        patch_sim.simulate_batch(neuron, protocols), test_voltages, protocols
    ):
        label = f"{voltage:+.0f} mV"
        color = SWEEP_COLORS[len(sweeps) % len(SWEEP_COLORS)]
        sweeps.append(Sweep.from_dataframe(df, protocol, label, color, VOLTAGE_CLAMP))

    return sweeps


def profile_build_figure(sweeps: list[Sweep], n_runs: int = 5) -> None:
    """Profile build_figure() over n_runs calls and print a cumulative report.

    Args:
        sweeps: Pre-built list of Sweep objects to pass to build_figure.
        n_runs: Number of times to call build_figure (averages out noise).
    """
    visibility = TraceVisibility()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_runs):
        build_figure(
            current_sweeps=sweeps,
            saved_sweeps=[],
            visibility=visibility,
            clamp_mode=VOLTAGE_CLAMP,
            stored_traces=None,
        )
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(buf.getvalue())


def main() -> None:
    """Run the profiling script end-to-end.

    Simulates the Na+ Channel Activation dataset then profiles the
    build_figure() call, printing a ranked summary to stdout.
    """
    print("Simulating Na+ Channel Activation dataset...")
    sweeps = build_sweeps()
    n_sweeps = len(sweeps)
    n_pts = len(sweeps[0].time) if sweeps else 0
    print(f"  {n_sweeps} sweeps × {n_pts} time points each\n")

    print("Profiling build_figure()...")
    profile_build_figure(sweeps)


if __name__ == "__main__":
    main()
