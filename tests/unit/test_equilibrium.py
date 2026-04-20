"""Tests for the equilibrium module."""

import pytest

from patch_sim import find_zero_current_voltage
from patch_sim.constants import PURKINJE, TRN
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
# PURKINJE is excluded: De Schutter & Bower (1994) Traub-Miles kinetics plus
# INaP/INaR produce a complex I_total(V) profile with multiple sign changes in
# the default range (−100 to −20 mV).  The pacemaker threshold at −65 mV is a
# genuine sign change, but Brent's method in the default range finds a spurious
# root near −30 mV (Na/K plateau region).  Use v_min=−75, v_max=−55 to isolate
# the physiological pacemaker threshold.
# See test_find_zero_current_voltage_purkinje_pacemaking_range below.
_EQUILIBRIUM_PRESET_NAMES = [p for p in NEURON_PRESET_NAMES if p not in (TRN, PURKINJE)]


@pytest.mark.parametrize("preset_name", _EQUILIBRIUM_PRESET_NAMES)
def test_find_zero_current_voltage_all_presets(preset_name: str) -> None:
    """Every non-TRN preset has a zero-current equilibrium that matches v_rest.

    After the v_rest fix, the declared v_rest should equal the computed
    equilibrium within 0.5 mV.
    """
    config = NEURON_PRESETS[preset_name]
    neuron = make_neuron(config)
    v_eq = find_zero_current_voltage(neuron)
    assert abs(v_eq - neuron.v_rest) < 0.5, (
        f"{preset_name}: computed equilibrium {v_eq:.2f} mV differs from "
        f"v_rest={neuron.v_rest:.1f} mV by {abs(v_eq - neuron.v_rest):.2f} mV"
    )


def test_find_zero_current_voltage_purkinje_pacemaking_range() -> None:
    """Purkinje zero-current equilibrium is in [−75, −55] mV and matches v_rest.

    With INaP/INaR added, the equilibrium shifts from −82.3 mV to the
    physiological pacemaking range (Häusser & Clark 1997: −55 to −65 mV).
    The default [−100, −20] range finds a spurious root near −30 mV (Na/K
    plateau); the narrow [−75, −55] range isolates the pacemaker threshold,
    pinning g_KL, g_NaP, g_NaR, and g_CaT against silent regressions.
    """
    config = NEURON_PRESETS[PURKINJE]
    neuron = make_neuron(config)
    v_eq = find_zero_current_voltage(neuron, v_min=-75.0, v_max=-55.0)
    assert abs(v_eq - neuron.v_rest) < 0.5, (
        f"Purkinje pacemaking-range equilibrium {v_eq:.2f} mV differs from "
        f"v_rest={neuron.v_rest:.1f} mV by {abs(v_eq - neuron.v_rest):.2f} mV"
    )
