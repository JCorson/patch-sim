"""Sanity-guard integration tests for dynamic E_Ca (issue #264 Phase 2).

Two guards:
1. A non-Ca preset produces a byte-identical trace — dynamic E_Ca plumbing
   must not affect channels without Ca²⁺ current.
2. A Ca preset under strong hyperpolarization (Ca activation suppressed)
   matches a frozen golden — verifies Phase 1 + Phase 2 together do not
   regress under the near-zero-Ca-current regime.
"""

import numpy as np

from patch_sim import simulate_current_clamp
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import DOPAMINERGIC, NEURON_PRESETS, PURKINJE

# ---------------------------------------------------------------------------
# Frozen goldens (every 10th sample of the full trace)
# Generated: 2026-04-24, commit 3c3628a (Phase 2 dynamic E_Ca complete)
# To regenerate: run scratch/find_coupled_equilibria.py and update these arrays.
# ---------------------------------------------------------------------------

_GOLDEN_DA_50MS_4UA = np.array(
    [
        -62.5,
        -61.53615527,
        -60.57499967,
        -59.53952912,
        -58.29003547,
        -56.43966456,
        -51.90243308,
        45.22167488,
        12.23702593,
        -41.66859012,
        -83.89944466,
        -87.520989,
        -86.16161635,
        -83.96229605,
        -81.66633229,
        -79.48733889,
        -77.48461445,
        -75.66608029,
        -74.02271676,
        -72.54050954,
        -71.20459388,
        -70.00063278,
        -68.91519592,
        -67.93577715,
        -67.05069206,
        -66.24895229,
        -65.52015389,
        -64.85438733,
        -64.24215876,
        -63.67430113,
        -63.14184571,
        -62.63581732,
        -62.14690064,
        -61.66488991,
        -61.17775274,
        -60.66994707,
        -60.11914291,
        -59.48911392,
        -58.71189965,
        -57.63239782,
        -55.7620688,
        -49.74750437,
        46.32199223,
        0.75004273,
        -52.42158271,
        -85.78202722,
        -87.67129666,
        -86.22394259,
        -84.17752099,
        -82.09192214,
        -80.12807389,
        -78.3277042,
        -76.6935069,
        -75.2157464,
        -73.88132212,
        -72.67684751,
        -71.58965208,
        -70.6080534,
        -69.72137336,
        -68.91987013,
        -68.19465013,
        -67.53758405,
        -66.94123511,
        -66.39880026,
        -65.90406205,
        -65.45134734,
        -65.03548922,
        -64.65178917,
        -64.29597738,
        -63.96417015,
        -63.65282394,
        -63.35868561,
        -63.07873877,
        -62.81014502,
        -62.55017852,
        -62.29615057,
        -62.04531866,
        -61.79477109,
        -61.54127211,
        -61.28104192,
        -61.00942609,
        -60.72036986,
        -60.40553042,
        -60.05267259,
        -59.6425244,
        -59.14194223,
        -58.48681363,
        -57.52946692,
        -55.80974231,
        -50.34780861,
        46.61758597,
        3.18660067,
        -49.92503542,
        -85.58721515,
        -87.85262747,
        -86.5283825,
        -84.57315636,
        -82.56820807,
        -80.67830154,
        -78.94599743,
        -77.37427169,
        -75.95363459,
        -74.67127665,
        -73.51414182,
        -72.46991701,
        -71.52729906,
        -70.67601199,
        -69.9067423,
        -69.21105325,
        -68.58130043,
        -68.01055592,
        -67.49254276,
        -67.0215787,
        -66.59252718,
        -66.20075379,
        -65.84208611,
        -65.51277604,
        -65.20946353,
        -64.92914161,
        -64.66912259,
        -64.42700554,
        -64.20064529,
        -63.98812319,
        -63.78771959,
        -63.59788831,
        -63.41723282,
        -63.2444842,
        -63.07848061,
        -62.91814788,
        -62.76248115,
        -62.61052678,
        -62.46136438,
        -62.31408798,
        -62.16778582,
        -62.02151742,
        -61.87428658,
        -61.72500816,
        -61.57246556,
        -61.41525436,
        -61.2517047,
        -61.0797713,
        -60.89687152,
        -60.69963882,
        -60.48353146,
        -60.24218211,
        -59.96625521,
        -59.64129726,
        -59.24332333,
        -58.72863738,
        -58.00614882,
        -56.84015002,
        -54.30095202,
        -33.14999792,
        33.69114821,
        -22.58241531,
        -75.22006581,
        -87.63641756,
        -87.47234289,
        -85.76141492,
        -83.78068405,
        -81.85065758,
        -80.06192704,
        -78.43294796,
        -76.95897167,
        -75.62834714,
        -74.42800113,
        -73.34524703,
        -72.36833291,
        -71.48655416,
        -70.6902187,
        -69.97056737,
        -69.31968633,
        -68.73042476,
        -68.19632161,
        -67.71154179,
        -67.27082047,
        -66.8694139,
        -66.50305521,
        -66.16791368,
        -65.86055691,
        -65.577915,
        -65.31724671,
        -65.07610742,
        -64.85231901,
        -64.64394167,
        -64.44924777,
        -64.26669777,
        -64.09491831,
        -63.93268214,
        -63.77889013,
        -63.63255501,
        -63.49278669,
        -63.35877916,
        -63.22979863,
        -63.10517278,
        -62.98428103,
        -62.86654544,
        -62.75142235,
        -62.63839428,
        -62.52696211,
        -62.41663717,
    ]
)

_GOLDEN_PC_20MS_NEG5UA = np.array(
    [
        -65.0,
        -65.80093327,
        -66.59501039,
        -67.39070298,
        -68.19153049,
        -68.99823537,
        -69.81001966,
        -70.62525423,
        -71.4418694,
        -72.25756985,
        -73.06995746,
        -73.87663082,
        -74.67525549,
        -75.46362952,
        -76.23974493,
        -77.00183785,
        -77.74842192,
        -78.47830178,
        -79.19056648,
        -79.88456605,
        -80.55987755,
        -81.21626578,
        -81.85363896,
        -82.47201273,
        -83.07148077,
        -83.65219169,
        -84.21433158,
        -84.75811158,
        -85.28375897,
        -85.79151106,
        -86.28161097,
        -86.75430471,
        -87.20983922,
        -87.64846102,
        -88.07041543,
        -88.47594604,
        -88.86529451,
        -89.23870054,
        -89.59640196,
        -89.93863499,
        -90.26563451,
        -90.57763438,
        -90.87486788,
        -91.15756804,
        -91.42596809,
        -91.6803018,
        -91.92080388,
        -92.14771037,
        -92.36125888,
        -92.56168897,
        -92.74924234,
        -92.92416308,
        -93.08669785,
        -93.23709599,
        -93.37560962,
        -93.50249368,
        -93.61800595,
        -93.722407,
        -93.81596009,
        -93.89893107,
        -93.97158823,
        -94.03420208,
        -94.08704513,
        -94.13039167,
        -94.16451743,
        -94.18969934,
        -94.20621517,
        -94.21434319,
        -94.21436184,
        -94.20654933,
        -94.19118333,
        -94.16854052,
        -94.1388963,
        -94.10252435,
        -94.05969631,
        -94.0106814,
        -93.95574607,
        -93.89515368,
        -93.82916418,
        -93.75803377,
        -93.68201463,
    ]
)


def _n_steps(duration_ms: float) -> int:
    """Return the number of simulation steps for a given duration.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Number of time steps including the initial point.
    """
    return int(duration_ms * SIM_SAMPLING_FREQ / 1000.0) + 1


def test_non_ca_preset_unaffected() -> None:
    """Dopaminergic (no Ca²⁺ channels) trace is byte-identical after Phase 2.

    The dynamic E_Ca plumbing passes ``ca_i`` to every ``compute_current``
    call, but for channels with ``carries_calcium=False`` (the only kind in
    the Dopaminergic preset) the argument is ignored.  The voltage trace must
    therefore be numerically identical to the Phase 1 golden.
    """
    neuron = make_neuron(NEURON_PRESETS[DOPAMINERGIC])
    assert neuron.calcium_dynamics is None, "Dopaminergic must have no Ca dynamics"

    n = _n_steps(50)
    stim = np.full(n, 4.0)
    result = simulate_current_clamp(neuron, current_external=stim)
    sampled = result["voltage"][::10]

    assert np.allclose(sampled, _GOLDEN_DA_50MS_4UA, atol=1e-10), (
        "Dopaminergic trace diverged from golden — dynamic E_Ca plumbing may "
        "have accidentally changed non-Ca channel computation."
    )


def test_hyperpolarising_ca_preset_minimal_change() -> None:
    """Purkinje under strong hyperpolarization matches golden within 1e-10 mV.

    Under −5 µA/cm² for 20 ms, Ca²⁺ activation gates close (d_inf → 0),
    so Ca²⁺ influx is suppressed and ca_i stays near ca_rest.  E_Ca
    therefore barely shifts from its resting value, and the voltage trace
    produced with dynamic E_Ca should match the frozen golden.
    """
    neuron = make_neuron(NEURON_PRESETS[PURKINJE])
    assert neuron.calcium_dynamics is not None, "Purkinje must have Ca dynamics"

    n = _n_steps(20)
    stim = np.full(n, -5.0)
    result = simulate_current_clamp(neuron, current_external=stim)
    sampled = result["voltage"][::10]

    assert np.allclose(sampled, _GOLDEN_PC_20MS_NEG5UA, atol=1e-10), (
        "Purkinje hyperpolarisation trace diverged from golden — check "
        "Ca²⁺ dynamics or E_Ca computation for regressions."
    )
