"""Tests for the core HH channel factory functions in core_channels.py."""

import pytest

from patch_sim.channels import IonChannel, IonSpecies, NernstSpec
from patch_sim.core_channels import (
    POSPISCHIL_VT,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    make_k_channel,
    make_leak_channel,
    make_na_channel,
    make_pospischil_k_channel,
    make_pospischil_na_channel,
    pospischil_alpha_h,
    pospischil_alpha_m,
    pospischil_alpha_n,
    pospischil_beta_h,
    pospischil_beta_m,
    pospischil_beta_n,
)
from patch_sim.neuron import Neuron

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_n_positive(V: float) -> None:
    """alpha_n is positive at physiological voltages."""
    assert alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_n_positive(V: float) -> None:
    """beta_n is positive at physiological voltages."""
    assert beta_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_m_positive(V: float) -> None:
    """alpha_m is positive at physiological voltages."""
    assert alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_m_positive(V: float) -> None:
    """beta_m is positive at physiological voltages."""
    assert beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_alpha_h_positive(V: float) -> None:
    """alpha_h is positive at physiological voltages."""
    assert alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_beta_h_positive(V: float) -> None:
    """beta_h is positive at physiological voltages."""
    assert beta_h(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards
# ---------------------------------------------------------------------------


def test_alpha_n_singularity_guard() -> None:
    """alpha_n returns 0.1 at the removable singularity V = −55 mV."""
    assert alpha_n(-55.0, 0.0) == pytest.approx(0.1)


def test_alpha_n_near_singularity_continuous_above() -> None:
    """alpha_n is continuous approaching −55 mV from above."""
    assert alpha_n(-55.0 + 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_n_near_singularity_continuous_below() -> None:
    """alpha_n is continuous approaching −55 mV from below."""
    assert alpha_n(-55.0 - 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_m_singularity_guard() -> None:
    """alpha_m returns 1.0 at the removable singularity V = −40 mV."""
    assert alpha_m(-40.0, 0.0) == pytest.approx(1.0)


def test_alpha_m_near_singularity_continuous_above() -> None:
    """alpha_m is continuous approaching −40 mV from above."""
    assert alpha_m(-40.0 + 1e-5, 0.0) == pytest.approx(1.0, rel=1e-3)


def test_alpha_m_near_singularity_continuous_below() -> None:
    """alpha_m is continuous approaching −40 mV from below."""
    assert alpha_m(-40.0 - 1e-5, 0.0) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize("fn", [alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h])
def test_rate_functions_ignore_ca_i(V: float, fn) -> None:
    """All rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_steady_state_gating_bounds(V: float) -> None:
    """Steady-state gating variables computed from module-level rates are in [0,1]."""
    n_inf = alpha_n(V, 0.0) / (alpha_n(V, 0.0) + beta_n(V, 0.0))
    m_inf = alpha_m(V, 0.0) / (alpha_m(V, 0.0) + beta_m(V, 0.0))
    h_inf = alpha_h(V, 0.0) / (alpha_h(V, 0.0) + beta_h(V, 0.0))
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_na_channel_structure() -> None:
    """make_na_channel returns a channel with correct name, gates, and reversal spec."""
    ch = make_na_channel(g_max=120.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(120.0)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_k_channel_structure() -> None:
    """make_k_channel returns a channel with correct name, gate, and reversal spec."""
    ch = make_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


def test_make_leak_channel_structure() -> None:
    """make_leak_channel returns a channel with no gates and Cl⁻ reversal spec."""
    ch = make_leak_channel(g_max=0.3)
    assert isinstance(ch, IonChannel)
    assert ch.name == "leak"
    assert ch.g_max == pytest.approx(0.3)
    assert len(ch.gating_variables) == 0
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.CHLORIDE
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Numerical equivalence with old inline formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -55.0, -40.0, 0.0, 20.0, 40.0])
def test_na_channel_current_matches_inline(V: float) -> None:
    """Na channel compute_current equals g_Na * m³ * h * (V − E_Na)."""
    neuron = Neuron()
    ch = make_na_channel(g_max=neuron.g_Na)

    m = alpha_m(V, 0.0) / (alpha_m(V, 0.0) + beta_m(V, 0.0))
    h = alpha_h(V, 0.0) / (alpha_h(V, 0.0) + beta_h(V, 0.0))
    gating_state = {"m": m, "h": h}

    result = ch.compute_current(V, gating_state, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = neuron.g_Na * (m**3) * h * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -55.0, 0.0, 20.0, 40.0])
def test_k_channel_current_matches_inline(V: float) -> None:
    """K channel compute_current equals g_K * n⁴ * (V − E_K)."""
    neuron = Neuron()
    ch = make_k_channel(g_max=neuron.g_K)

    n = alpha_n(V, 0.0) / (alpha_n(V, 0.0) + beta_n(V, 0.0))
    gating_state = {"n": n}

    result = ch.compute_current(V, gating_state, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = neuron.g_K * (n**4) * (V - E_K)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_leak_channel_current_matches_inline(V: float) -> None:
    """Leak channel compute_current equals g_L * (V − E_L)."""
    neuron = Neuron()
    ch = make_leak_channel(g_max=neuron.g_L)

    result = ch.compute_current(V, {}, neuron)
    E_L = ch.reversal_potential(neuron)
    expected = neuron.g_L * (V - E_L)
    assert result == pytest.approx(expected)


# ===========================================================================
# Pospischil et al. (2008) cortical RS kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_m_positive(V: float) -> None:
    """pospischil_alpha_m is positive at physiological voltages."""
    assert pospischil_alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_m_positive(V: float) -> None:
    """pospischil_beta_m is positive at physiological voltages."""
    assert pospischil_beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_h_positive(V: float) -> None:
    """pospischil_alpha_h is positive at physiological voltages."""
    assert pospischil_alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_h_positive(V: float) -> None:
    """pospischil_beta_h is positive at physiological voltages."""
    assert pospischil_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_alpha_n_positive(V: float) -> None:
    """pospischil_alpha_n is positive at physiological voltages."""
    assert pospischil_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_beta_n_positive(V: float) -> None:
    """pospischil_beta_n is positive at physiological voltages."""
    assert pospischil_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards
# ---------------------------------------------------------------------------

# Singularity voltages derived from VT = -56.2 mV:
#   pospischil_alpha_m: V = VT + 13 = -43.2 mV, limit = 0.32 * 4 = 1.28
#   pospischil_beta_m:  V = VT + 40 = -16.2 mV, limit = 0.28 * 5 = 1.4
#   pospischil_alpha_n: V = VT + 15 = -41.2 mV, limit = 0.032 * 5 = 0.16

_ALPHA_M_SINGULARITY = POSPISCHIL_VT + 13  # -43.2 mV
_BETA_M_SINGULARITY = POSPISCHIL_VT + 40  # -16.2 mV
_ALPHA_N_SINGULARITY = POSPISCHIL_VT + 15  # -41.2 mV


def test_pospischil_alpha_m_singularity_guard() -> None:
    """pospischil_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_pospischil_alpha_m_near_singularity_continuous_above() -> None:
    """pospischil_alpha_m is continuous approaching the singularity from above."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_pospischil_alpha_m_near_singularity_continuous_below() -> None:
    """pospischil_alpha_m is continuous approaching the singularity from below."""
    assert pospischil_alpha_m(_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_pospischil_beta_m_singularity_guard() -> None:
    """pospischil_beta_m returns L'Hôpital limit 1.4 at V = VT + 40."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_pospischil_beta_m_near_singularity_continuous_above() -> None:
    """pospischil_beta_m is continuous approaching the singularity from above."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_pospischil_beta_m_near_singularity_continuous_below() -> None:
    """pospischil_beta_m is continuous approaching the singularity from below."""
    assert pospischil_beta_m(_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_pospischil_alpha_n_singularity_guard() -> None:
    """pospischil_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_pospischil_alpha_n_near_singularity_continuous_above() -> None:
    """pospischil_alpha_n is continuous approaching the singularity from above."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_pospischil_alpha_n_near_singularity_continuous_below() -> None:
    """pospischil_alpha_n is continuous approaching the singularity from below."""
    assert pospischil_alpha_n(_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        pospischil_alpha_m,
        pospischil_beta_m,
        pospischil_alpha_h,
        pospischil_beta_h,
        pospischil_alpha_n,
        pospischil_beta_n,
    ],
)
def test_pospischil_rate_functions_ignore_ca_i(V: float, fn) -> None:
    """All Pospischil rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_pospischil_steady_state_gating_bounds(V: float) -> None:
    """Pospischil steady-state gating variables are in [0, 1]."""
    n_inf = pospischil_alpha_n(V, 0.0) / (
        pospischil_alpha_n(V, 0.0) + pospischil_beta_n(V, 0.0)
    )
    m_inf = pospischil_alpha_m(V, 0.0) / (
        pospischil_alpha_m(V, 0.0) + pospischil_beta_m(V, 0.0)
    )
    h_inf = pospischil_alpha_h(V, 0.0) / (
        pospischil_alpha_h(V, 0.0) + pospischil_beta_h(V, 0.0)
    )
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_pospischil_na_channel_structure() -> None:
    """make_pospischil_na_channel returns a channel with correct structure."""
    ch = make_pospischil_na_channel(g_max=50.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(50.0)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_pospischil_k_channel_structure() -> None:
    """make_pospischil_k_channel returns a channel with correct structure."""
    ch = make_pospischil_k_channel(g_max=5.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(5.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Numerical equivalence with inline formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -43.2, -16.2, 0.0, 20.0, 40.0])
def test_pospischil_na_channel_current_matches_inline(V: float) -> None:
    """Pospischil Na channel compute_current equals g_Na * m³ * h * (V − E_Na)."""
    neuron = Neuron()
    ch = make_pospischil_na_channel(g_max=neuron.g_Na)

    m = pospischil_alpha_m(V, 0.0) / (
        pospischil_alpha_m(V, 0.0) + pospischil_beta_m(V, 0.0)
    )
    h = pospischil_alpha_h(V, 0.0) / (
        pospischil_alpha_h(V, 0.0) + pospischil_beta_h(V, 0.0)
    )
    gating_state = {"m": m, "h": h}

    result = ch.compute_current(V, gating_state, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = neuron.g_Na * (m**3) * h * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -80.0, -65.0, -41.2, 0.0, 20.0, 40.0])
def test_pospischil_k_channel_current_matches_inline(V: float) -> None:
    """Pospischil K channel compute_current equals g_K * n⁴ * (V − E_K)."""
    neuron = Neuron()
    ch = make_pospischil_k_channel(g_max=neuron.g_K)

    n = pospischil_alpha_n(V, 0.0) / (
        pospischil_alpha_n(V, 0.0) + pospischil_beta_n(V, 0.0)
    )
    gating_state = {"n": n}

    result = ch.compute_current(V, gating_state, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = neuron.g_K * (n**4) * (V - E_K)
    assert result == pytest.approx(expected)
