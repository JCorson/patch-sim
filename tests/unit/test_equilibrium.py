"""Tests for the equilibrium module."""

import pytest

from patch_sim import find_zero_current_voltage
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


@pytest.mark.parametrize("preset_name", NEURON_PRESET_NAMES)
def test_find_zero_current_voltage_all_presets(preset_name: str) -> None:
    """Every preset has a zero-current equilibrium, and it matches v_rest.

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
