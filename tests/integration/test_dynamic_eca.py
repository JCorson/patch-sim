"""Sanity-guard integration tests for dynamic E_Ca (issue #264 Phase 2).

Two property-based guards:

1. A non-Ca preset's voltage trace is invariant to ``Neuron.Ca_in`` —
   proving the dynamic E_Ca plumbing does not accidentally couple non-Ca
   channels to the intracellular Ca²⁺ concentration.
2. Under strong hyperpolarization, a Ca-carrying preset's ``ca_i`` trace
   stays pinned to ``ca_rest`` within tens of nanomolar — proving Ca²⁺
   activation collapses (and therefore Ca²⁺ influx is suppressed) when
   expected.
"""

import dataclasses

import numpy as np

from patch_sim import simulate_current_clamp
from patch_sim.constants import SIM_SAMPLING_FREQ
from patch_sim.presets import CORTICAL_PYRAMIDAL, NEURON_PRESETS, PURKINJE


def _n_steps(duration_ms: float) -> int:
    """Return the number of simulation steps for a given duration.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Number of time steps including the initial point.
    """
    return int(duration_ms * SIM_SAMPLING_FREQ / 1000.0) + 1


def test_non_ca_preset_invariant_to_ca_in() -> None:
    """Cortical Pyramidal V trace is byte-identical under a 100× Ca_in perturbation.

    Cortical Pyramidal carries no Ca²⁺ channels and has ``calcium_dynamics=None``,
    so ``Neuron.Ca_in`` should never enter any ``compute_current`` call.
    Running the simulation with ``Ca_in`` scaled by 100 must yield a
    byte-identical voltage trace; otherwise the dynamic E_Ca plumbing has
    accidentally exposed ``Ca_in`` to a channel that should ignore it.
    """
    neuron_default = NEURON_PRESETS[CORTICAL_PYRAMIDAL]()
    neuron_perturbed = dataclasses.replace(
        neuron_default, Ca_in=neuron_default.Ca_in * 100
    )
    assert neuron_default.calcium_dynamics is None, (
        "Cortical Pyramidal must have no Ca dynamics for this invariance to hold"
    )

    n = _n_steps(50)
    stim = np.full(n, 4.0)
    v_default = simulate_current_clamp(neuron_default, current_external=stim)["voltage"]
    v_perturbed = simulate_current_clamp(neuron_perturbed, current_external=stim)[
        "voltage"
    ]

    np.testing.assert_array_equal(
        v_default,
        v_perturbed,
        err_msg=(
            "Non-Ca preset trace changed when Ca_in was perturbed — dynamic "
            "E_Ca plumbing may be reading ca_i in a channel that should "
            "ignore it."
        ),
    )


def test_hyperpolarization_suppresses_ca_influx() -> None:
    """Strong Purkinje hyperpolarization keeps ca_i pinned to ca_rest.

    Under −5 µA/cm² for 20 ms, V drops toward E_K (~−95 mV), the Ca²⁺
    activation gates (d for ICaL, ICaT) close, and Ca²⁺ influx vanishes.
    The simulated ``ca_i`` should therefore remain within a few tens of
    nanomolar of ``ca_rest`` for the entire run — orders of magnitude
    smaller than the 0.1 µM physiological lower band asserted by
    :mod:`tests.integration.test_calcium_calibration`.  A regression that
    re-enables Ca²⁺ influx at rest would break this bound by 10× or more.
    """
    neuron = NEURON_PRESETS[PURKINJE]()
    cd = neuron.calcium_dynamics
    assert cd is not None, "Purkinje must have Ca dynamics"

    n = _n_steps(20)
    stim = np.full(n, -5.0)
    result = simulate_current_clamp(neuron, current_external=stim)

    assert "ca_i" in result.dtype.names
    max_dev_mm = float(np.max(np.abs(result["ca_i"] - cd.ca_rest)))
    # Empirically ~1.1e-5 mM (11 nM) on the calibrated Phase 2 build; 5e-5 mM
    # leaves ~5× headroom for benign tweaks while still catching any
    # regression that re-couples Ca²⁺ activation to rest.
    assert max_dev_mm < 5e-5, (
        f"ca_i deviated {max_dev_mm:.2e} mM from ca_rest under strong "
        "hyperpolarization — Ca²⁺ activation may not be properly suppressed."
    )
