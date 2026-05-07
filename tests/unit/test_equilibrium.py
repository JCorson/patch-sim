"""Tests for the equilibrium module."""

import warnings

import pytest

from patch_sim import find_coupled_equilibrium, find_zero_current_voltage
from patch_sim.constants import (
    CORTICAL_PYRAMIDAL,
    DOPAMINERGIC,
    PURKINJE,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.equilibrium import _total_ionic_current
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESET_NAMES, NEURON_PRESETS


def test_total_ionic_current_type(hh_model: Neuron) -> None:
    """_total_ionic_current returns a float."""
    result = _total_ionic_current(hh_model, -65.0, 0.0)
    assert isinstance(result, float)


def test_find_zero_current_voltage_default_neuron(hh_model: Neuron) -> None:
    """Default Neuron() has a zero-current equilibrium within the search range."""
    v_eq = find_zero_current_voltage(hh_model)
    assert -100.0 < v_eq < -20.0


def test_find_zero_current_voltage_near_zero_current(hh_model: Neuron) -> None:
    """Total ionic current is near zero at the computed equilibrium voltage."""
    v_eq = find_zero_current_voltage(hh_model)
    ca_i = 0.0
    i_total = _total_ionic_current(hh_model, v_eq, ca_i)
    assert abs(i_total) < 1e-4


def test_find_zero_current_voltage_invalid_range() -> None:
    """ValueError is raised when v_min >= v_max."""
    neuron = Neuron()
    with pytest.raises(ValueError, match="v_min"):
        find_zero_current_voltage(neuron, v_min=-50.0, v_max=-80.0)


def test_find_zero_current_voltage_no_bracket() -> None:
    """ValueError is raised when the range does not bracket a root."""
    neuron = Neuron()
    # A range entirely below any physiological resting potential
    with pytest.raises(ValueError, match="do not bracket a root"):
        find_zero_current_voltage(neuron, v_min=-100.0, v_max=-95.0)


# TRN is excluded: Huguenard & Prince kinetics (VT=−67 mV) + ICaT window
# current create an I_total(V) profile that approaches zero near −80 mV but
# without a sign change detectable by Brent's method in the default search
# range (−100 to −20 mV).  The only clear sign change in [−100, −20] is the
# Na/K balance at −37 mV, not the physiological rest at −80 mV.
# The resting potential is maintained dynamically (ICaT window current balanced
# by mixed Na⁺/K⁺ leak), but does not manifest as a Brent-findable zero
# crossing in the default search range.  See test_all_presets_stable_at_rest
# in test_current_clamp.py for the 50 ms stability verification.
#
# PURKINJE is excluded: with Ih added the cell is an autonomous oscillator
# with NO stable zero-current equilibrium anywhere in the physiological range.
# Ih (half-activation ≈ −82 mV) provides a large inward current at all
# subthreshold voltages, so I_total has no sign change.  v_rest = −65 mV is
# the simulation starting point (near the INaP threshold), not an equilibrium.
# See test_fires_spontaneously in tests/integration/test_purkinje.py.
#
# DOPAMINERGIC is excluded: Canavier/Komendantov kinetics (VT=−67 mV) combined
# with large Ih (g_max=2.0) create multiple sign changes in the default search
# range [−100, −20] mV — one near the physiological rest (~−62.5 mV) and at
# least one more near the Na⁺ activation threshold (~−37 mV).  Brent's method
# finds whichever root the endpoint signs bracket; with the default endpoints it
# converges on the non-physiological root near −37 mV rather than the resting
# equilibrium.  Stability at v_rest is verified instead by the simulation-based
# test_all_presets_stable_at_rest in tests/integration/test_current_clamp.py.
#
# CORTICAL_PYRAMIDAL is excluded: after the #311 fix replaced the Pospischil
# delayed rectifier with the slow Mainen-Sejnowski Kv (closed at rest, opens
# only above ~0 mV) as the sole K conductance, the cell has no active outward
# current at depolarised voltages.  Pospischil Na's window current drives
# I_total(V=-20 mV) negative, so the default bracket [-100, -20] no longer
# spans a sign change at the resting equilibrium near -70 mV (additional
# zero crossings appear between -65 and -50 mV).  The cell rests stably at
# -70 mV in simulation, verified by test_no_spontaneous_firing in
# tests/integration/test_cortical_pyramidal.py and test_all_presets_stable_at_rest.
#
# STN is excluded post-#305: the autonomous-pacemaker preset uses INaP and Ih
# to destabilise rest near −60 mV, and the Mainen-Sejnowski Kv replaces the
# Otsuka K factory as the sole delayed rectifier (closed at rest).  I_total
# has no Brent-findable root in the default [−100, −20] mV bracket — the cell
# fires spontaneously from v_rest = −60 mV.  See test_stn_spontaneous_pacemaking
# in tests/integration/test_stn.py and test_all_presets_stable_at_rest.
_EQUILIBRIUM_PRESET_NAMES = [
    p
    for p in NEURON_PRESET_NAMES
    if p not in (TRN, PURKINJE, DOPAMINERGIC, CORTICAL_PYRAMIDAL, STN)
]


@pytest.mark.parametrize("preset_name", _EQUILIBRIUM_PRESET_NAMES)
def test_find_zero_current_voltage_all_presets(preset_name: str) -> None:
    """Every non-TRN, non-Purkinje preset has a coupled equilibrium near v_rest.

    Uses :func:`find_coupled_equilibrium` so that Ca²⁺-carrying presets with
    window currents at rest are tested against the dynamically-coupled (V, ca_i)
    equilibrium, which is what ``v_rest`` is calibrated to.  For non-Ca presets
    the coupled and static equilibria are identical.

    The declared v_rest should equal the computed equilibrium within 0.5 mV.
    """
    config = NEURON_PRESETS[preset_name]
    neuron = make_neuron(config)
    v_eq, _ = find_coupled_equilibrium(neuron)
    assert abs(v_eq - neuron.v_rest) < 0.5, (
        f"{preset_name}: computed equilibrium {v_eq:.4f} mV differs from "
        f"v_rest={neuron.v_rest:.4f} mV by {abs(v_eq - neuron.v_rest):.4f} mV"
    )


def test_find_coupled_equilibrium_warns_on_non_convergence() -> None:
    """RuntimeWarning is emitted when max_iter elapses without convergence.

    Uses the THALAMIC_RELAY preset, which has window-current-driven coupling
    between V and ca_i and therefore requires several fixed-point iterations
    to converge.  Setting max_iter=1 ensures convergence cannot be reached.
    """
    neuron = make_neuron(NEURON_PRESETS[THALAMIC_RELAY])
    with pytest.warns(RuntimeWarning, match="did not converge"):
        v_eq, ca_i_eq = find_coupled_equilibrium(neuron, max_iter=1)
    assert isinstance(v_eq, float)
    assert ca_i_eq > 0.0


def test_find_coupled_equilibrium_no_warning_on_convergence() -> None:
    """No warning is emitted when iteration converges within max_iter."""
    neuron = make_neuron(NEURON_PRESETS[THALAMIC_RELAY])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        find_coupled_equilibrium(neuron)
