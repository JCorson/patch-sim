"""Regression tests for the voltage-clamp rate-table optimization (#261).

Compares the default (tabled) path against a forced-scalar fallback path
obtained by monkeypatching :func:`_precompute_rate_tables` to return all-None
entries.  The two paths must produce numerically identical results across
HH-only, Purkinje (Boltzmann/cosh rates), and SNc Dopaminergic (IKCa +
calcium dynamics) presets.
"""

import numpy as np
import pytest

from patch_sim import clamp_simulations
from patch_sim.channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from patch_sim.clamp_simulations import (
    _is_voltage_only,
    _precompute_rate_tables,
    simulate_voltage_clamp,
)
from patch_sim.constants import DOPAMINERGIC, PURKINJE
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS


def _force_scalar_tables(
    neuron: Neuron, voltage_protocol: np.ndarray
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None]]:
    """Return all-None rate tables so the RK4 loop evaluates every rate scalar-wise.

    Args:
        neuron: The neuron whose gating variables are being tabulated.
        voltage_protocol: Prescribed voltage array (unused; present for signature).

    Returns:
        Two dicts keyed by gating-variable name, every value ``None``.
    """
    del voltage_protocol  # signature parity with _precompute_rate_tables
    alpha_tables: dict[str, np.ndarray | None] = {
        gv.name: None for gv in neuron.all_gating_variables
    }
    beta_tables: dict[str, np.ndarray | None] = {
        gv.name: None for gv in neuron.all_gating_variables
    }
    return alpha_tables, beta_tables


def _step_protocol(num_steps: int = 12_000) -> np.ndarray:
    """Build a short hold → step → hold voltage protocol.

    Args:
        num_steps: Total length of the protocol in samples.

    Returns:
        Voltage array (mV) exercising pre-step, step, and post-step behaviour.
    """
    protocol = np.full(num_steps, -80.0)
    step_start = num_steps // 3
    step_end = 2 * num_steps // 3
    protocol[step_start:step_end] = 0.0
    return protocol


@pytest.mark.parametrize(
    "neuron_factory",
    [
        pytest.param(lambda: Neuron(), id="hh-default"),
        pytest.param(
            lambda: make_neuron(NEURON_PRESETS[PURKINJE]),
            id="purkinje",
        ),
        pytest.param(
            lambda: make_neuron(NEURON_PRESETS[DOPAMINERGIC]),
            id="snc-dopaminergic",
        ),
    ],
)
def test_tabled_and_scalar_paths_agree(monkeypatch, neuron_factory):
    """Tabled VC output must match the forced-scalar path bit-for-bit.

    Monkeypatches :func:`_precompute_rate_tables` to emit all-None tables,
    forcing the RK4 loop to evaluate every rate scalar-wise.  Compares against
    the default tabled path for every field in the structured result.
    """
    neuron = neuron_factory()
    protocol = _step_protocol()

    tabled = simulate_voltage_clamp(neuron, protocol)

    monkeypatch.setattr(
        clamp_simulations, "_precompute_rate_tables", _force_scalar_tables
    )
    scalar = simulate_voltage_clamp(neuron, protocol)

    assert tabled.dtype.names == scalar.dtype.names
    assert tabled.dtype.names is not None
    for field in tabled.dtype.names:
        np.testing.assert_allclose(
            tabled[field],
            scalar[field],
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"Field {field!r} diverges between tabled and scalar paths",
        )


def test_is_voltage_only_detects_custom_ca_dependent_rate():
    """A hand-rolled Ca²⁺-dependent rate must probe as Ca-dependent.

    Exercises the probe path (non-:class:`BoltzmannCoshRate` callable) with
    a callable whose output is a linear function of ca_i, ensuring
    ``_precompute_rate_tables`` emits ``None`` for such gates.
    """

    def ca_linear_alpha(V: float, ca_i: float) -> float:
        """Linear-in-ca_i rate (alpha-like).

        Args:
            V: Membrane voltage in mV (unused).
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            A value that varies with ca_i.
        """
        del V
        return 0.1 + 10.0 * ca_i

    def ca_linear_beta(V: float, ca_i: float) -> float:
        """Another linear-in-ca_i rate (beta-like).

        Args:
            V: Membrane voltage in mV (unused).
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            A value that varies with ca_i.
        """
        del V
        return 0.5 + 5.0 * ca_i

    assert _is_voltage_only(ca_linear_alpha) is False
    assert _is_voltage_only(ca_linear_beta) is False

    custom_gate = GatingVariable(
        name="ca_probe",
        power=1,
        alpha=ca_linear_alpha,
        beta=ca_linear_beta,
    )
    custom_channel = IonChannel(
        name="CaProbe",
        g_max=0.0,
        gating_variables=(custom_gate,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(additional_channels=(custom_channel,))

    protocol = _step_protocol(num_steps=200)
    alpha_tables, beta_tables = _precompute_rate_tables(neuron, protocol)

    assert alpha_tables["ca_probe"] is None
    assert beta_tables["ca_probe"] is None


def test_is_voltage_only_handles_raising_rate():
    """A rate that raises must be classified Ca-dependent (conservative).

    This keeps the optimization safe for rate functions with unusual
    signatures that happen to reject the probe's Ca²⁺ values.
    """

    def raising_rate(V: float, ca_i: float) -> float:
        """Rate function that always raises.

        Args:
            V: Membrane voltage in mV (unused).
            ca_i: Intracellular Ca²⁺ concentration in mM (unused).

        Raises:
            RuntimeError: Always.
        """
        del V, ca_i
        raise RuntimeError("probe failure")

    assert _is_voltage_only(raising_rate) is False
