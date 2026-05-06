"""Tests for the core HH channel factory functions in core_channels.py."""

import math

import pytest

from patch_sim.channels import IonChannel, IonSpecies, NernstSpec
from patch_sim.core_channels import (
    DOPAMINERGIC_VT,
    MAINEN_SEJNOWSKI_KV_PRESCALE,
    MAINEN_SEJNOWSKI_KV_VHALF,
    POSPISCHIL_VT,
    PURKINJE_VT,
    _pospischil_alpha_sNa,
    _pospischil_beta_sNa,
    _purkinje_alpha_sNa,
    _purkinje_beta_sNa,
    _stn_alpha_h,
    _stn_alpha_m,
    _stn_alpha_n,
    _stn_alpha_sNa,
    _stn_beta_h,
    _stn_beta_m,
    _stn_beta_n,
    _stn_beta_sNa,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    dopaminergic_alpha_h,
    dopaminergic_alpha_m,
    dopaminergic_alpha_n,
    dopaminergic_beta_h,
    dopaminergic_beta_m,
    dopaminergic_beta_n,
    mainen_sejnowski_alpha_h,
    mainen_sejnowski_alpha_m,
    mainen_sejnowski_alpha_n,
    mainen_sejnowski_beta_h,
    mainen_sejnowski_beta_m,
    mainen_sejnowski_beta_n,
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_k_channel,
    make_k_leak_channel,
    make_mainen_sejnowski_kv_channel,
    make_mainen_sejnowski_na_channel,
    make_na_channel,
    make_na_leak_channel,
    make_pospischil_k_channel,
    make_pospischil_na_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
    make_stn_k_channel,
    make_stn_na_channel,
    pospischil_alpha_h,
    pospischil_alpha_m,
    pospischil_alpha_n,
    pospischil_beta_h,
    pospischil_beta_m,
    pospischil_beta_n,
    purkinje_alpha_h,
    purkinje_alpha_m,
    purkinje_alpha_n,
    purkinje_beta_h,
    purkinje_beta_m,
    purkinje_beta_n,
)
from patch_sim.neuron import Neuron
from patch_sim.rates import Rate

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
    """alpha_n is finite and positive at the singularity V = −55 mV."""
    val = alpha_n(-55.0, 0.0)
    assert math.isfinite(val)
    assert val > 0.0


def test_alpha_n_near_singularity_continuous_above() -> None:
    """alpha_n is continuous approaching −55 mV from above."""
    assert alpha_n(-55.0 + 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_n_near_singularity_continuous_below() -> None:
    """alpha_n is continuous approaching −55 mV from below."""
    assert alpha_n(-55.0 - 1e-5, 0.0) == pytest.approx(0.1, rel=1e-3)


def test_alpha_m_singularity_guard() -> None:
    """alpha_m is finite and positive at the singularity V = −40 mV."""
    val = alpha_m(-40.0, 0.0)
    assert math.isfinite(val)
    assert val > 0.0


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
def test_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
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


def test_make_na_leak_channel_structure() -> None:
    """make_na_leak_channel returns a channel with no gates and Na⁺ reversal spec."""
    ch = make_na_leak_channel(g_max=0.054)
    assert isinstance(ch, IonChannel)
    assert ch.name == "NaL"
    assert ch.g_max == pytest.approx(0.054)
    assert len(ch.gating_variables) == 0
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_k_leak_channel_structure() -> None:
    """make_k_leak_channel returns a channel with no gates and K⁺ reversal spec."""
    ch = make_k_leak_channel(g_max=0.246)
    assert isinstance(ch, IonChannel)
    assert ch.name == "KL"
    assert ch.g_max == pytest.approx(0.246)
    assert len(ch.gating_variables) == 0
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
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
def test_na_leak_channel_current_matches_inline(V: float) -> None:
    """Na leak channel compute_current equals g_NaL * (V − E_Na)."""
    neuron = Neuron()
    ch = make_na_leak_channel(g_max=neuron.g_NaL)

    result = ch.compute_current(V, {}, neuron)
    E_Na = ch.reversal_potential(neuron)
    expected = neuron.g_NaL * (V - E_Na)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_k_leak_channel_current_matches_inline(V: float) -> None:
    """K leak channel compute_current equals g_KL * (V − E_K)."""
    neuron = Neuron()
    ch = make_k_leak_channel(g_max=neuron.g_KL)

    result = ch.compute_current(V, {}, neuron)
    E_K = ch.reversal_potential(neuron)
    expected = neuron.g_KL * (V - E_K)
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
def test_pospischil_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
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


def test_make_pospischil_na_channel_with_slow_inactivation() -> None:
    """``slow_inactivation=True`` adds the sNa gate (Fleidervish & Gutnick 1996)."""
    ch = make_pospischil_na_channel(g_max=35.0, slow_inactivation=True)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[2].name == "sNa"
    assert ch.gating_variables[2].power == 1


# ---------------------------------------------------------------------------
# Pospischil slow Na inactivation rate functions (sNa gate; Fleidervish &
# Gutnick 1996; Mickus, Jung & Spruston 1999).  Opt-in gate added in #327
# to abolish the residual depol-block plateau that single-gate Pospischil
# kinetics leave open under sustained suprathreshold drive.
# ---------------------------------------------------------------------------


def _pospischil_sNa_inf(V: float) -> float:
    """Steady-state availability of the Pospischil slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa gate at voltage V, in [0, 1].
    """
    a, b = _pospischil_alpha_sNa(V, 0.0), _pospischil_beta_sNa(V, 0.0)
    return a / (a + b)


def test_pospischil_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _pospischil_alpha_sNa(V, 0.0)
        b = _pospischil_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa negative at V={V}"
        assert b >= 0, f"beta_sNa negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa steady state {ss} out of [0,1] at V={V}"


def test_pospischil_slow_na_inactivation_decreases_with_depolarisation() -> None:
    """The sNa availability decreases monotonically with depolarisation."""
    assert (
        _pospischil_sNa_inf(-80.0)
        > _pospischil_sNa_inf(-50.0)
        > _pospischil_sNa_inf(-15.0)
    )


def test_pospischil_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa sits at -50 mV (Fleidervish & Gutnick 1996 mid-range)."""
    assert _pospischil_sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_pospischil_slow_na_inactivation_resting_availability() -> None:
    """Cortical pyramidal rests at -70 mV — sNa must remain near-fully open."""
    # Cortical pyramidal v_rest = -70 mV (deeper than STN's -60 mV cycle), so
    # subthreshold sNa availability should be even higher than the STN gate's
    # rest value.  Loss of >10 % rest availability would noticeably suppress
    # AP amplitude on every step from rest.
    assert _pospischil_sNa_inf(-70.0) > 0.9


def test_pospischil_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarised plateau voltages sNa closes, abolishing the residual h-tail."""
    # The depol-block plateau the new gate must escape (mirroring #324) hangs
    # at ≈ −15 mV; sNa must close firmly there so g_Na_eff = g_max * m^3 * h
    # * sNa collapses below the leak + IM outward drive.
    assert _pospischil_sNa_inf(-15.0) < 0.05


def test_pospischil_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa at V½ stays distinctly slow vs the fast m, h gates."""
    a = _pospischil_alpha_sNa(-50.0, 0.0)
    b = _pospischil_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa tau at V½ is {tau:.1f} ms, expected > 100 ms"


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


# ===========================================================================
# Mainen & Sejnowski (1996) cortical pyramidal Kv kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 25.0, 40.0])
def test_mainen_sejnowski_alpha_n_positive(V: float) -> None:
    """mainen_sejnowski_alpha_n is positive at physiological voltages."""
    assert mainen_sejnowski_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 25.0, 40.0])
def test_mainen_sejnowski_beta_n_positive(V: float) -> None:
    """mainen_sejnowski_beta_n is positive at physiological voltages."""
    assert mainen_sejnowski_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards at V = V_½ = +25 mV
# ---------------------------------------------------------------------------

# Expected L'Hôpital limits with the 23 → 34 °C Q10 = 2.3 pre-scale baked in:
#   alpha_n(25) = 0.02 * 2.55 * 9 = 0.459 / ms
#   beta_n(25)  = 0.002 * 2.55 * 9 = 0.0459 / ms
_MS_ALPHA_N_LIMIT = 0.02 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
_MS_BETA_N_LIMIT = 0.002 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0


def test_mainen_sejnowski_alpha_n_singularity_guard() -> None:
    """mainen_sejnowski_alpha_n returns the L'Hôpital limit at V = 25 mV."""
    assert mainen_sejnowski_alpha_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0) == pytest.approx(
        _MS_ALPHA_N_LIMIT
    )


def test_mainen_sejnowski_alpha_n_continuous_above() -> None:
    """mainen_sejnowski_alpha_n is continuous approaching the singularity from above."""
    assert mainen_sejnowski_alpha_n(
        MAINEN_SEJNOWSKI_KV_VHALF + 1e-5, 0.0
    ) == pytest.approx(_MS_ALPHA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_alpha_n_continuous_below() -> None:
    """mainen_sejnowski_alpha_n is continuous approaching the singularity from below."""
    assert mainen_sejnowski_alpha_n(
        MAINEN_SEJNOWSKI_KV_VHALF - 1e-5, 0.0
    ) == pytest.approx(_MS_ALPHA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_beta_n_singularity_guard() -> None:
    """mainen_sejnowski_beta_n returns the L'Hôpital limit at V = 25 mV."""
    assert mainen_sejnowski_beta_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0) == pytest.approx(
        _MS_BETA_N_LIMIT
    )


def test_mainen_sejnowski_beta_n_continuous_above() -> None:
    """mainen_sejnowski_beta_n is continuous approaching the singularity from above."""
    assert mainen_sejnowski_beta_n(
        MAINEN_SEJNOWSKI_KV_VHALF + 1e-5, 0.0
    ) == pytest.approx(_MS_BETA_N_LIMIT, rel=1e-3)


def test_mainen_sejnowski_beta_n_continuous_below() -> None:
    """mainen_sejnowski_beta_n is continuous approaching the singularity from below."""
    assert mainen_sejnowski_beta_n(
        MAINEN_SEJNOWSKI_KV_VHALF - 1e-5, 0.0
    ) == pytest.approx(_MS_BETA_N_LIMIT, rel=1e-3)


# ---------------------------------------------------------------------------
# Steady-state shape: high-threshold, mostly closed at rest
# ---------------------------------------------------------------------------


def test_mainen_sejnowski_n_inf_closed_at_rest() -> None:
    """n_inf is < 0.05 at v_rest = -70 mV (channel almost fully closed)."""
    a = mainen_sejnowski_alpha_n(-70.0, 0.0)
    b = mainen_sejnowski_beta_n(-70.0, 0.0)
    n_inf = a / (a + b)
    assert n_inf < 0.05


def test_mainen_sejnowski_n_inf_strongly_open_at_peak() -> None:
    """n_inf > 0.9 above the AP peak voltage so K⁺ repolarises after the spike.

    The published M-S Kv has α/β = 10 at V = ``MAINEN_SEJNOWSKI_KV_VHALF`` (the
    rate-function singularity), so n_inf ≈ 0.91 there.  This is intentional —
    the channel is essentially fully open above ~+10 mV, providing strong
    repolarising drive once the AP threshold is crossed.
    """
    a = mainen_sejnowski_alpha_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0)
    b = mainen_sejnowski_beta_n(MAINEN_SEJNOWSKI_KV_VHALF, 0.0)
    n_inf = a / (a + b)
    assert n_inf > 0.9


def test_mainen_sejnowski_n_inf_half_activated_near_zero() -> None:
    """n_inf passes through 0.5 between -5 and +10 mV (true activation V_½).

    Solving α(V) = β(V) for the published prefactors gives V_½ ≈ +4.3 mV.
    The activation V_½ is *not* the same as MAINEN_SEJNOWSKI_KV_VHALF — that
    constant marks the rate-function singularity at V = +25 mV.
    """
    n_inf_minus5 = mainen_sejnowski_alpha_n(-5.0, 0.0) / (
        mainen_sejnowski_alpha_n(-5.0, 0.0) + mainen_sejnowski_beta_n(-5.0, 0.0)
    )
    n_inf_plus10 = mainen_sejnowski_alpha_n(10.0, 0.0) / (
        mainen_sejnowski_alpha_n(10.0, 0.0) + mainen_sejnowski_beta_n(10.0, 0.0)
    )
    assert n_inf_minus5 < 0.5 < n_inf_plus10


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0, 30.0])
@pytest.mark.parametrize(
    "fn",
    [mainen_sejnowski_alpha_n, mainen_sejnowski_beta_n],
)
def test_mainen_sejnowski_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """Mainen-Sejnowski Kv rates return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_mainen_sejnowski_kv_channel_structure() -> None:
    """make_mainen_sejnowski_kv_channel returns a channel with correct structure.

    Uses channel name ``"Kv"`` (not ``"K"``) and gating-variable name ``"nKv"``
    so it can coexist with a Pospischil delayed rectifier in the same neuron
    if a future preset wants a dual-K architecture.
    """
    ch = make_mainen_sejnowski_kv_channel(g_max=30.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Kv"
    assert ch.g_max == pytest.approx(30.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "nKv"
    assert ch.gating_variables[0].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Mainen-Sejnowski Na rate functions and factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -35.0, 0.0, 40.0])
def test_mainen_sejnowski_na_rates_positive(V: float) -> None:
    """All M-S Na rate functions are positive at physiological voltages."""
    assert mainen_sejnowski_alpha_m(V, 0.0) > 0
    assert mainen_sejnowski_beta_m(V, 0.0) > 0
    assert mainen_sejnowski_alpha_h(V, 0.0) > 0
    assert mainen_sejnowski_beta_h(V, 0.0) > 0


def test_mainen_sejnowski_alpha_m_singularity_guard() -> None:
    """alpha_m returns the trap0 L'Hôpital limit at V = tha = -35 mV."""
    expected = 0.182 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    assert mainen_sejnowski_alpha_m(-35.0, 0.0) == pytest.approx(expected)


def test_mainen_sejnowski_beta_m_singularity_guard() -> None:
    """beta_m hits its singularity at the same V = -35 mV (-V = -tha branch).

    The na.mod ``trap0(-v, -tha, Rb, qa)`` form has its removable singularity
    at -V = -tha = +35 mV, i.e. V = -35 mV — the same physical voltage as
    the α_m singularity.
    """
    expected = 0.124 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    assert mainen_sejnowski_beta_m(-35.0, 0.0) == pytest.approx(expected)


def test_mainen_sejnowski_alpha_m_continuous_around_singularity() -> None:
    """alpha_m is continuous approaching the singularity from either side."""
    expected = 0.182 * MAINEN_SEJNOWSKI_KV_PRESCALE * 9.0
    above = mainen_sejnowski_alpha_m(-35.0 + 1e-5, 0.0)
    below = mainen_sejnowski_alpha_m(-35.0 - 1e-5, 0.0)
    assert above == pytest.approx(expected, rel=1e-3)
    assert below == pytest.approx(expected, rel=1e-3)


def test_mainen_sejnowski_h_inf_open_at_rest() -> None:
    """h_inf > 0.5 at v_rest = -70 mV (Na channels mostly available for the next AP).

    Boltzmann form gives h_inf(-70) = 1/(1 + exp(-5/6.2)) ≈ 0.69 — just
    above half-availability, consistent with most Na channels being open
    and ready to fire.
    """
    a = mainen_sejnowski_alpha_h(-70.0, 0.0)
    b = mainen_sejnowski_beta_h(-70.0, 0.0)
    h_inf = a / (a + b)
    assert h_inf > 0.5


def test_mainen_sejnowski_h_inf_strongly_inactivated_at_minus_20() -> None:
    """h_inf < 0.001 at V = -20 mV (steady-state Na window current is tiny).

    Critical for the cortical pyramidal preset's equilibrium analysis:
    Pospischil Na has h_inf ≈ 0.034 at -20 mV, producing a large Na window
    current that prevents the bracket [-100, -20] from finding a unique
    zero crossing once Kv kinetics are slowed.  M-S Na's stronger
    inactivation (h_inf ≈ 7e-4 at -20 mV) eliminates that current.
    """
    a = mainen_sejnowski_alpha_h(-20.0, 0.0)
    b = mainen_sejnowski_beta_h(-20.0, 0.0)
    h_inf = a / (a + b)
    assert h_inf < 0.001


def test_mainen_sejnowski_m_inf_strongly_open_at_peak() -> None:
    """m_inf > 0.95 at V = +30 mV (full activation during AP peak)."""
    a = mainen_sejnowski_alpha_m(30.0, 0.0)
    b = mainen_sejnowski_beta_m(30.0, 0.0)
    m_inf = a / (a + b)
    assert m_inf > 0.95


@pytest.mark.parametrize("V", [-65.0, 0.0, 30.0])
@pytest.mark.parametrize(
    "fn",
    [
        mainen_sejnowski_alpha_m,
        mainen_sejnowski_beta_m,
        mainen_sejnowski_alpha_h,
        mainen_sejnowski_beta_h,
    ],
)
def test_mainen_sejnowski_na_rates_ignore_ca_i(V: float, fn: Rate) -> None:
    """M-S Na rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


def test_make_mainen_sejnowski_na_channel_structure() -> None:
    """make_mainen_sejnowski_na_channel returns a channel with correct structure."""
    ch = make_mainen_sejnowski_na_channel(g_max=50.0)
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


# ---------------------------------------------------------------------------
# Otsuka et al. (2004) STN channel kinetics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -40.0, 0.0, 40.0])
def test_stn_na_rate_functions_positive(V: float) -> None:
    """All STN Na⁺ rate functions are strictly positive at physiological voltages."""
    assert _stn_alpha_m(V, 0.0) > 0
    assert _stn_beta_m(V, 0.0) > 0
    assert _stn_alpha_h(V, 0.0) > 0
    assert _stn_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -41.0, 0.0, 40.0])
def test_stn_k_rate_functions_positive(V: float) -> None:
    """All STN K⁺ DR rate functions are strictly positive at physiological voltages."""
    assert _stn_alpha_n(V, 0.0) > 0
    assert _stn_beta_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -40.0, 0.0, 40.0])
def test_stn_na_steady_state_bounds(V: float) -> None:
    """STN Na⁺ steady-state gating variables are in [0, 1]."""
    m_inf = _stn_alpha_m(V, 0.0) / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0))
    h_inf = _stn_alpha_h(V, 0.0) / (_stn_alpha_h(V, 0.0) + _stn_beta_h(V, 0.0))
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


@pytest.mark.parametrize("V", [-100.0, -65.0, -41.0, 0.0, 40.0])
def test_stn_k_steady_state_bounds(V: float) -> None:
    """STN K⁺ DR steady-state gating variable is in [0, 1]."""
    n_inf = _stn_alpha_n(V, 0.0) / (_stn_alpha_n(V, 0.0) + _stn_beta_n(V, 0.0))
    assert 0.0 <= n_inf <= 1.0


@pytest.mark.parametrize(
    "V, expected_m_inf",
    [
        (-80.0, 1.0 / (1.0 + math.exp(-(-80.0 + 40.0) / 8.0))),
        (-40.0, 0.5),  # V_half of m
        (0.0, 1.0 / (1.0 + math.exp(-(0.0 + 40.0) / 8.0))),
    ],
)
def test_stn_na_m_inf_matches_boltzmann(V: float, expected_m_inf: float) -> None:
    """STN Na⁺ m steady-state matches 1/(1+exp(-(V+40)/8))."""
    m_inf = _stn_alpha_m(V, 0.0) / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0))
    assert m_inf == pytest.approx(expected_m_inf, rel=1e-6)


@pytest.mark.parametrize(
    "V, expected_h_inf",
    [
        (-80.0, 1.0 / (1.0 + math.exp((-80.0 + 45.5) / 6.4))),
        (-45.5, 0.5),  # V_half of h
        (-20.0, 1.0 / (1.0 + math.exp((-20.0 + 45.5) / 6.4))),
    ],
)
def test_stn_na_h_inf_matches_boltzmann(V: float, expected_h_inf: float) -> None:
    """STN Na⁺ h steady-state matches 1/(1+exp((V+45.5)/6.4))."""
    h_inf = _stn_alpha_h(V, 0.0) / (_stn_alpha_h(V, 0.0) + _stn_beta_h(V, 0.0))
    assert h_inf == pytest.approx(expected_h_inf, rel=1e-6)


@pytest.mark.parametrize(
    "V, expected_n_inf",
    [
        (-80.0, 1.0 / (1.0 + math.exp(-(-80.0 + 41.0) / 14.0))),
        (-41.0, 0.5),  # V_half of n
        (0.0, 1.0 / (1.0 + math.exp(-(0.0 + 41.0) / 14.0))),
    ],
)
def test_stn_k_n_inf_matches_boltzmann(V: float, expected_n_inf: float) -> None:
    """STN K⁺ n steady-state matches 1/(1+exp(-(V+41)/14))."""
    n_inf = _stn_alpha_n(V, 0.0) / (_stn_alpha_n(V, 0.0) + _stn_beta_n(V, 0.0))
    assert n_inf == pytest.approx(expected_n_inf, rel=1e-6)


def test_stn_na_m_tau_is_voltage_independent() -> None:
    """STN Na⁺ activation time constant is positive and voltage-independent."""
    voltages = (-80.0, -40.0, 0.0, 40.0)
    taus = [1.0 / (_stn_alpha_m(V, 0.0) + _stn_beta_m(V, 0.0)) for V in voltages]
    assert all(tau > 0.0 for tau in taus)
    assert all(tau == pytest.approx(taus[0], rel=1e-6) for tau in taus)


def test_make_stn_na_channel_structure() -> None:
    """make_stn_na_channel returns correct gates and Na⁺ reversal spec."""
    ch = make_stn_na_channel(g_max=49.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(49.0)
    assert len(ch.gating_variables) == 2
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


def test_make_stn_na_channel_with_slow_inactivation() -> None:
    """``slow_inactivation=True`` adds the sNa gate (Fleidervish & Gutnick 1996)."""
    ch = make_stn_na_channel(g_max=30.0, slow_inactivation=True)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[2].name == "sNa"
    assert ch.gating_variables[2].power == 1


# ---------------------------------------------------------------------------
# STN slow Na inactivation rate functions (sNa gate; Fleidervish & Gutnick
# 1996; Mickus, Jung & Spruston 1999; Do & Bean 2003).  Opt-in gate added in
# #324 to abolish the residual −15 mV plateau the Otsuka 2004 fast h-tail
# leaves open.
# ---------------------------------------------------------------------------


def _sNa_inf(V: float) -> float:
    """Steady-state availability of the STN slow Na inactivation gate."""
    a, b = _stn_alpha_sNa(V, 0.0), _stn_beta_sNa(V, 0.0)
    return a / (a + b)


def test_stn_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _stn_alpha_sNa(V, 0.0)
        b = _stn_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa negative at V={V}"
        assert b >= 0, f"beta_sNa negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa steady state {ss} out of [0,1] at V={V}"


def test_stn_slow_na_inactivation_decreases_with_depolarisation() -> None:
    """The sNa availability decreases monotonically with depolarisation."""
    assert _sNa_inf(-80.0) > _sNa_inf(-50.0) > _sNa_inf(-15.0)


def test_stn_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa sits at -50 mV (Fleidervish & Gutnick 1996 mid-range)."""
    assert _sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_stn_slow_na_inactivation_resting_availability() -> None:
    """Near-rest sNa availability stays high to preserve autonomous pacemaking."""
    # STN cycles around -60 mV at the autonomous rate; sNa must remain high
    # there or the spike train would lose Na availability.
    assert _sNa_inf(-65.0) > 0.85
    assert _sNa_inf(-75.0) > 0.93


def test_stn_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarised plateau voltages sNa closes, abolishing the residual h-tail."""
    # The depol-block plateau in #324 hung at ≈ −15 mV; sNa must close
    # firmly there so g_Na_eff = g_max * m^3 * h * sNa collapses.
    assert _sNa_inf(-15.0) < 0.05


def test_stn_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa at V½ stays distinctly slow vs the fast m, h gates."""
    a, b = _stn_alpha_sNa(-50.0, 0.0), _stn_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa tau at V½ is {tau:.1f} ms, expected > 100 ms"


def test_make_stn_k_channel_structure() -> None:
    """make_stn_k_channel returns a channel with correct gate and K⁺ reversal spec."""
    ch = make_stn_k_channel(g_max=57.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(57.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


def test_stn_preset_uses_otsuka_factories() -> None:
    """STN preset uses make_stn_na_channel and make_stn_k_channel factories."""
    import functools

    from patch_sim.constants import STN
    from patch_sim.presets import NEURON_PRESETS

    config = NEURON_PRESETS[STN]
    # The Na factory is wrapped in functools.partial(..., slow_inactivation=True)
    # to opt into the sNa gate (#324 depol-block recovery).
    assert isinstance(config.na_channel_factory, functools.partial)
    assert config.na_channel_factory.func is make_stn_na_channel
    assert config.na_channel_factory.keywords == {"slow_inactivation": True}
    assert config.k_channel_factory is make_stn_k_channel


# ---------------------------------------------------------------------------
# Purkinje (De Schutter & Bower 1994) rate functions
# ---------------------------------------------------------------------------

_PURKINJE_ALPHA_M_SINGULARITY = PURKINJE_VT + 13  # -45 mV
_PURKINJE_BETA_M_SINGULARITY = PURKINJE_VT + 40  # -18 mV
_PURKINJE_ALPHA_N_SINGULARITY = PURKINJE_VT + 15  # -43 mV


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_na_rate_functions_positive(V: float) -> None:
    """All Purkinje Na⁺ rate functions are positive at physiological voltages."""
    assert purkinje_alpha_m(V, 0.0) > 0
    assert purkinje_beta_m(V, 0.0) > 0
    assert purkinje_alpha_h(V, 0.0) > 0
    assert purkinje_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_k_rate_functions_positive(V: float) -> None:
    """All Purkinje K⁺ rate functions are positive at physiological voltages."""
    assert purkinje_alpha_n(V, 0.0) > 0
    assert purkinje_beta_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_purkinje_steady_state_gating_bounds(V: float) -> None:
    """Purkinje steady-state gating variables are in [0, 1]."""
    m_inf = purkinje_alpha_m(V, 0.0) / (
        purkinje_alpha_m(V, 0.0) + purkinje_beta_m(V, 0.0)
    )
    h_inf = purkinje_alpha_h(V, 0.0) / (
        purkinje_alpha_h(V, 0.0) + purkinje_beta_h(V, 0.0)
    )
    n_inf = purkinje_alpha_n(V, 0.0) / (
        purkinje_alpha_n(V, 0.0) + purkinje_beta_n(V, 0.0)
    )
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0
    assert 0.0 <= n_inf <= 1.0


def test_purkinje_alpha_m_singularity_guard() -> None:
    """purkinje_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_purkinje_alpha_m_near_singularity_continuous_above() -> None:
    """purkinje_alpha_m is continuous approaching the singularity from above."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_purkinje_alpha_m_near_singularity_continuous_below() -> None:
    """purkinje_alpha_m is continuous approaching the singularity from below."""
    assert purkinje_alpha_m(_PURKINJE_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_purkinje_beta_m_singularity_guard() -> None:
    """purkinje_beta_m returns L'Hôpital limit 1.4 at V = VT + 40."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_purkinje_beta_m_near_singularity_continuous_above() -> None:
    """purkinje_beta_m is continuous approaching the singularity from above."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_purkinje_beta_m_near_singularity_continuous_below() -> None:
    """purkinje_beta_m is continuous approaching the singularity from below."""
    assert purkinje_beta_m(_PURKINJE_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_purkinje_alpha_n_singularity_guard() -> None:
    """purkinje_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_purkinje_alpha_n_near_singularity_continuous_above() -> None:
    """purkinje_alpha_n is continuous approaching the singularity from above."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_purkinje_alpha_n_near_singularity_continuous_below() -> None:
    """purkinje_alpha_n is continuous approaching the singularity from below."""
    assert purkinje_alpha_n(_PURKINJE_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        purkinje_alpha_m,
        purkinje_beta_m,
        purkinje_alpha_h,
        purkinje_beta_h,
        purkinje_alpha_n,
        purkinje_beta_n,
    ],
)
def test_purkinje_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All Purkinje rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


def test_make_purkinje_na_channel_structure() -> None:
    """make_purkinje_na_channel returns m³, h, and sNa gates with Na⁺ reversal."""
    ch = make_purkinje_na_channel(g_max=120.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "Na"
    assert ch.g_max == pytest.approx(120.0)
    assert len(ch.gating_variables) == 3
    assert ch.gating_variables[0].name == "m"
    assert ch.gating_variables[0].power == 3
    assert ch.gating_variables[1].name == "h"
    assert ch.gating_variables[1].power == 1
    assert ch.gating_variables[2].name == "sNa"
    assert ch.gating_variables[2].power == 1
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.SODIUM
    assert not ch.carries_calcium


# ---------------------------------------------------------------------------
# Purkinje slow Na inactivation rate functions (sNa gate; Carter & Bean
# 2009).  Always-on gate added in #329 to abolish the residual depol-block
# plateau under sustained climbing-fibre-style drive.
# ---------------------------------------------------------------------------


def _purkinje_sNa_inf(V: float) -> float:
    """Steady-state availability of the Purkinje slow Na inactivation gate.

    Args:
        V: Membrane voltage in mV.

    Returns:
        Steady-state availability of the sNa gate at voltage V, in [0, 1].
    """
    a, b = _purkinje_alpha_sNa(V, 0.0), _purkinje_beta_sNa(V, 0.0)
    return a / (a + b)


def test_purkinje_slow_na_inactivation_steady_state_in_bounds() -> None:
    """The sNa rates are positive and steady state in [0, 1] across V."""
    for V in (-120.0, -100.0, -75.0, -65.0, -50.0, -30.0, -15.0, 0.0, 30.0):
        a = _purkinje_alpha_sNa(V, 0.0)
        b = _purkinje_beta_sNa(V, 0.0)
        assert a >= 0, f"alpha_sNa negative at V={V}"
        assert b >= 0, f"beta_sNa negative at V={V}"
        ss = a / (a + b)
        assert 0.0 <= ss <= 1.0, f"sNa steady state {ss} out of [0,1] at V={V}"


def test_purkinje_slow_na_inactivation_decreases_with_depolarisation() -> None:
    """The sNa availability decreases monotonically with depolarisation."""
    assert (
        _purkinje_sNa_inf(-80.0) > _purkinje_sNa_inf(-50.0) > _purkinje_sNa_inf(-15.0)
    )


def test_purkinje_slow_na_inactivation_half_voltage() -> None:
    """V½ for sNa sits at -50 mV (mirrors STN / Pospischil mid-range)."""
    assert _purkinje_sNa_inf(-50.0) == pytest.approx(0.5, abs=0.01)


def test_purkinje_slow_na_inactivation_resting_availability() -> None:
    """Purkinje rests near -65 mV — sNa must remain mostly open."""
    # Purkinje v_rest = -65 mV (matches the STN cycle hyperpolarised end).
    # Loss of >15 % rest availability would noticeably suppress AP amplitude
    # on every spontaneous beat.
    assert _purkinje_sNa_inf(-65.0) > 0.85


def test_purkinje_slow_na_inactivation_blocks_depol_plateau() -> None:
    """At depolarised plateau voltages sNa closes, abolishing the residual h-tail."""
    # The depol-block plateau the new gate must escape (mirroring #324) hangs
    # at ≈ −15 mV; sNa must close firmly there so g_Na_eff = g_max * m^3 * h
    # * sNa collapses below the leak + IK outward drive.
    assert _purkinje_sNa_inf(-15.0) < 0.05


def test_purkinje_slow_na_inactivation_tau_is_slow() -> None:
    """τ_sNa at V½ stays distinctly slow vs the fast m, h gates."""
    a = _purkinje_alpha_sNa(-50.0, 0.0)
    b = _purkinje_beta_sNa(-50.0, 0.0)
    tau = 1.0 / (a + b)
    assert tau > 100.0, f"sNa tau at V½ is {tau:.1f} ms, expected > 100 ms"


def test_make_purkinje_k_channel_structure() -> None:
    """make_purkinje_k_channel returns a channel with n⁴ gating and K⁺ reversal."""
    ch = make_purkinje_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium


# ===========================================================================
# Komendantov et al. (2004) SNc dopaminergic (VT = -67 mV) kinetics
# ===========================================================================

# ---------------------------------------------------------------------------
# Rate function positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, -54.0, 0.0, 40.0])
def test_dopaminergic_alpha_m_positive(V: float) -> None:
    """dopaminergic_alpha_m is positive at physiological voltages and singularity."""
    assert dopaminergic_alpha_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -27.0, 0.0, 40.0])
def test_dopaminergic_beta_m_positive(V: float) -> None:
    """dopaminergic_beta_m is positive at physiological voltages and singularity."""
    assert dopaminergic_beta_m(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_alpha_h_positive(V: float) -> None:
    """dopaminergic_alpha_h is positive at physiological voltages."""
    assert dopaminergic_alpha_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_beta_h_positive(V: float) -> None:
    """dopaminergic_beta_h is positive at physiological voltages."""
    assert dopaminergic_beta_h(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, -52.0, 0.0, 40.0])
def test_dopaminergic_alpha_n_positive(V: float) -> None:
    """dopaminergic_alpha_n is positive at physiological voltages and singularity."""
    assert dopaminergic_alpha_n(V, 0.0) > 0


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_beta_n_positive(V: float) -> None:
    """dopaminergic_beta_n is positive at physiological voltages."""
    assert dopaminergic_beta_n(V, 0.0) > 0


# ---------------------------------------------------------------------------
# Singularity guards  (VT = -67 mV → singularities at -54, -27, -52 mV)
# ---------------------------------------------------------------------------

_DA_ALPHA_M_SINGULARITY = DOPAMINERGIC_VT + 13  # -54 mV; limit = 0.32 * 4 = 1.28
_DA_BETA_M_SINGULARITY = DOPAMINERGIC_VT + 40  # -27 mV; limit = 0.28 * 5 = 1.4
_DA_ALPHA_N_SINGULARITY = DOPAMINERGIC_VT + 15  # -52 mV; limit = 0.032 * 5 = 0.16


def test_dopaminergic_alpha_m_singularity_guard() -> None:
    """dopaminergic_alpha_m returns L'Hôpital limit 1.28 at V = VT + 13 = -54 mV."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY, 0.0) == pytest.approx(1.28)


def test_dopaminergic_alpha_m_near_singularity_continuous_above() -> None:
    """dopaminergic_alpha_m is continuous approaching -54 mV from above."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_dopaminergic_alpha_m_near_singularity_continuous_below() -> None:
    """dopaminergic_alpha_m is continuous approaching -54 mV from below."""
    assert dopaminergic_alpha_m(_DA_ALPHA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.28, rel=1e-3
    )


def test_dopaminergic_beta_m_singularity_guard() -> None:
    """dopaminergic_beta_m returns L'Hôpital limit 1.4 at V = VT + 40 = -27 mV."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY, 0.0) == pytest.approx(1.4)


def test_dopaminergic_beta_m_near_singularity_continuous_above() -> None:
    """dopaminergic_beta_m is continuous approaching -27 mV from above."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_dopaminergic_beta_m_near_singularity_continuous_below() -> None:
    """dopaminergic_beta_m is continuous approaching -27 mV from below."""
    assert dopaminergic_beta_m(_DA_BETA_M_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        1.4, rel=1e-3
    )


def test_dopaminergic_alpha_n_singularity_guard() -> None:
    """dopaminergic_alpha_n returns L'Hôpital limit 0.16 at V = VT + 15 = -52 mV."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY, 0.0) == pytest.approx(0.16)


def test_dopaminergic_alpha_n_near_singularity_continuous_above() -> None:
    """dopaminergic_alpha_n is continuous approaching -52 mV from above."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY + 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


def test_dopaminergic_alpha_n_near_singularity_continuous_below() -> None:
    """dopaminergic_alpha_n is continuous approaching -52 mV from below."""
    assert dopaminergic_alpha_n(_DA_ALPHA_N_SINGULARITY - 1e-5, 0.0) == pytest.approx(
        0.16, rel=1e-3
    )


# ---------------------------------------------------------------------------
# ca_i independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-65.0, 0.0])
@pytest.mark.parametrize(
    "fn",
    [
        dopaminergic_alpha_m,
        dopaminergic_beta_m,
        dopaminergic_alpha_h,
        dopaminergic_beta_h,
        dopaminergic_alpha_n,
        dopaminergic_beta_n,
    ],
)
def test_dopaminergic_rate_functions_ignore_ca_i(V: float, fn: Rate) -> None:
    """All dopaminergic rate functions return the same value regardless of ca_i."""
    assert fn(V, 0.0) == pytest.approx(fn(V, 1.0))


# ---------------------------------------------------------------------------
# Steady-state bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [-100.0, -65.0, 0.0, 40.0])
def test_dopaminergic_steady_state_gating_bounds(V: float) -> None:
    """Dopaminergic steady-state gating variables are in [0, 1]."""
    n_inf = dopaminergic_alpha_n(V, 0.0) / (
        dopaminergic_alpha_n(V, 0.0) + dopaminergic_beta_n(V, 0.0)
    )
    m_inf = dopaminergic_alpha_m(V, 0.0) / (
        dopaminergic_alpha_m(V, 0.0) + dopaminergic_beta_m(V, 0.0)
    )
    h_inf = dopaminergic_alpha_h(V, 0.0) / (
        dopaminergic_alpha_h(V, 0.0) + dopaminergic_beta_h(V, 0.0)
    )
    assert 0.0 <= n_inf <= 1.0
    assert 0.0 <= m_inf <= 1.0
    assert 0.0 <= h_inf <= 1.0


# ---------------------------------------------------------------------------
# Factory function structure
# ---------------------------------------------------------------------------


def test_make_dopaminergic_na_channel_structure() -> None:
    """make_dopaminergic_na_channel returns a Na⁺ channel with m³h gating."""
    ch = make_dopaminergic_na_channel(g_max=120.0)
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


def test_make_dopaminergic_k_channel_structure() -> None:
    """make_dopaminergic_k_channel returns a channel with n⁴ gating and K⁺ reversal."""
    ch = make_dopaminergic_k_channel(g_max=36.0)
    assert isinstance(ch, IonChannel)
    assert ch.name == "K"
    assert ch.g_max == pytest.approx(36.0)
    assert len(ch.gating_variables) == 1
    assert ch.gating_variables[0].name == "n"
    assert ch.gating_variables[0].power == 4
    assert isinstance(ch.reversal_spec, NernstSpec)
    assert ch.reversal_spec.species is IonSpecies.POTASSIUM
    assert not ch.carries_calcium
