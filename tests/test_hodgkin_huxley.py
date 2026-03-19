"""Tests for the core Hodgkin-Huxley model functionality."""

import dataclasses
import pytest
from ap_sim.channels import IonSpecies
from ap_sim.hodgkin_huxley import HodgkinHuxley
from ap_sim.electrochemistry import nernst_potential


def test_initialization(hh_model):
    """Test that the model is initialized with correct parameters."""
    assert hh_model.C_m == pytest.approx(1.0)
    assert hh_model.g_Na == pytest.approx(120.0)
    assert hh_model.g_K == pytest.approx(36.0)
    assert hh_model.g_L == pytest.approx(0.3)

    # Test that reversal potentials are within expected ranges
    assert 60.0 < hh_model.E_Na < 65.0
    assert -90.0 < hh_model.E_K < -65.0
    assert -70.0 < hh_model.E_L < -40.0


@pytest.mark.parametrize("voltage", [-100.0, -65.0, 0.0, 40.0])
def test_rate_constants(hh_model, voltage: float):
    """Test that all rate constants are positive at a range of membrane voltages."""
    assert hh_model.alpha_n(voltage) > 0
    assert hh_model.beta_n(voltage) > 0
    assert hh_model.alpha_m(voltage) > 0
    assert hh_model.beta_m(voltage) > 0
    assert hh_model.alpha_h(voltage) > 0
    assert hh_model.beta_h(voltage) > 0


@pytest.mark.parametrize("voltage", [-100.0, -65.0, 0.0, 40.0])
def test_steady_state_gating_bounds(hh_model, voltage: float):
    """Test that steady-state gating variables stay in [0, 1] at any voltage."""
    alpha_n = hh_model.alpha_n(voltage)
    beta_n = hh_model.beta_n(voltage)
    n_inf = alpha_n / (alpha_n + beta_n)

    alpha_m = hh_model.alpha_m(voltage)
    beta_m = hh_model.beta_m(voltage)
    m_inf = alpha_m / (alpha_m + beta_m)

    alpha_h = hh_model.alpha_h(voltage)
    beta_h = hh_model.beta_h(voltage)
    h_inf = alpha_h / (alpha_h + beta_h)

    assert 0 <= n_inf <= 1
    assert 0 <= m_inf <= 1
    assert 0 <= h_inf <= 1


def test_steady_state_values(hh_model):
    """Test that steady-state gating variables have correct resting-potential values."""
    voltage = -65.0

    # Calculate steady-state values
    alpha_n = hh_model.alpha_n(voltage)
    beta_n = hh_model.beta_n(voltage)
    n_inf = alpha_n / (alpha_n + beta_n)

    alpha_m = hh_model.alpha_m(voltage)
    beta_m = hh_model.beta_m(voltage)
    m_inf = alpha_m / (alpha_m + beta_m)

    alpha_h = hh_model.alpha_h(voltage)
    beta_h = hh_model.beta_h(voltage)
    h_inf = alpha_h / (alpha_h + beta_h)

    # At resting potential, h should be high, m and n should be low
    assert h_inf > 0.5  # Sodium inactivation high at rest
    assert m_inf < 0.5  # Sodium activation low at rest
    assert n_inf < 0.5  # Potassium activation low at rest


def test_time_constants(hh_model):
    """Test that time constants are positive and reasonable."""
    voltage = -65.0

    # Calculate time constants
    tau_n = 1.0 / (hh_model.alpha_n(voltage) + hh_model.beta_n(voltage))
    tau_m = 1.0 / (hh_model.alpha_m(voltage) + hh_model.beta_m(voltage))
    tau_h = 1.0 / (hh_model.alpha_h(voltage) + hh_model.beta_h(voltage))

    # Time constants should be positive
    assert tau_n > 0
    assert tau_m > 0
    assert tau_h > 0

    # Time constants should be in reasonable physiological range (ms)
    assert 0.1 <= tau_n <= 10
    assert 0.01 <= tau_m <= 1
    assert 1 <= tau_h <= 10


def test_custom_initialization():
    """Test that the model can be initialized with custom parameters."""
    custom_g_Na = 100.0
    custom_model = HodgkinHuxley(g_Na=custom_g_Na)

    assert custom_model.g_Na == pytest.approx(custom_g_Na)

    # Other parameters should still have default values
    assert custom_model.C_m == pytest.approx(1.0)
    assert custom_model.g_K == pytest.approx(36.0)
    assert custom_model.g_L == pytest.approx(0.3)


def test_singularity_guards(hh_model):
    """Test the near-singularity guards in alpha_n and alpha_m."""
    # alpha_n has a removable singularity at V = -55 mV
    # The limit as V -> -55 is 0.1
    result = hh_model.alpha_n(-55.0)
    assert result == pytest.approx(0.1)

    # Values just outside the guard threshold should be continuous with the limit
    # — approach from above
    result_near_above = hh_model.alpha_n(-55.0 + 1e-5)
    assert result_near_above == pytest.approx(0.1, rel=1e-3)

    # — approach from below
    result_near_below = hh_model.alpha_n(-55.0 - 1e-5)
    assert result_near_below == pytest.approx(0.1, rel=1e-3)

    # alpha_m has a removable singularity at V = -40 mV
    # The limit as V -> -40 is 1.0
    result = hh_model.alpha_m(-40.0)
    assert result == pytest.approx(1.0)

    # Values just outside the guard threshold should be continuous with the limit
    # — approach from above
    result_near_above = hh_model.alpha_m(-40.0 + 1e-5)
    assert result_near_above == pytest.approx(1.0, rel=1e-3)

    # — approach from below
    result_near_below = hh_model.alpha_m(-40.0 - 1e-5)
    assert result_near_below == pytest.approx(1.0, rel=1e-3)


def test_frozen_immutability(hh_model):
    """Assigning to a frozen dataclass field must raise FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        hh_model.g_Na = 999.0  # type: ignore[misc]


def test_reversal_potentials_match_nernst(hh_model):
    """E_Na, E_K, E_L must match Nernst equation for default concentrations."""
    expected_E_Na = nernst_potential(1, hh_model.T, hh_model.Na_out, hh_model.Na_in)
    expected_E_K = nernst_potential(1, hh_model.T, hh_model.K_out, hh_model.K_in)
    expected_E_L = nernst_potential(-1, hh_model.T, hh_model.Cl_out, hh_model.Cl_in)

    assert hh_model.E_Na == pytest.approx(expected_E_Na)
    assert hh_model.E_K == pytest.approx(expected_E_K)
    assert hh_model.E_L == pytest.approx(expected_E_L)


def test_custom_ion_concentrations_shift_reversal_potentials():
    """Changing ion concentrations must produce shifted reversal potentials."""
    custom_model = HodgkinHuxley(Na_out=200.0, K_in=100.0, T=293.15)
    default_model = HodgkinHuxley()

    # Higher extracellular Na+ → more positive E_Na
    assert custom_model.E_Na > default_model.E_Na

    # Lower intracellular K+ (100 vs 140 mM) → K_out/K_in ratio is larger
    # → more positive E_K
    assert custom_model.E_K > default_model.E_K

    # Values must still agree with direct Nernst calculation
    assert custom_model.E_Na == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.Na_out, custom_model.Na_in)
    )
    assert custom_model.E_K == pytest.approx(
        nernst_potential(1, custom_model.T, custom_model.K_out, custom_model.K_in)
    )


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g_Na", [-1.0, -0.001])
def test_negative_g_Na_raises(g_Na: float):
    """Negative sodium conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_Na"):
        HodgkinHuxley(g_Na=g_Na)


@pytest.mark.parametrize("g_K", [-1.0, -0.001])
def test_negative_g_K_raises(g_K: float):
    """Negative potassium conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_K"):
        HodgkinHuxley(g_K=g_K)


@pytest.mark.parametrize("g_L", [-1.0, -0.001])
def test_negative_g_L_raises(g_L: float):
    """Negative leak conductance must raise ValueError."""
    with pytest.raises(ValueError, match="g_L"):
        HodgkinHuxley(g_L=g_L)


@pytest.mark.parametrize("C_m", [0.0, -1.0])
def test_non_positive_capacitance_raises(C_m: float):
    """Non-positive membrane capacitance must raise ValueError."""
    with pytest.raises(ValueError, match="C_m"):
        HodgkinHuxley(C_m=C_m)


@pytest.mark.parametrize("T", [0.0, -1.0])
def test_non_positive_temperature_raises(T: float):
    """Non-positive temperature must raise ValueError."""
    with pytest.raises(ValueError, match="Temperature"):
        HodgkinHuxley(T=T)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Na_out": 0.0},
        {"Na_out": -1.0},
        {"Na_in": 0.0},
        {"Na_in": -1.0},
        {"K_out": 0.0},
        {"K_out": -1.0},
        {"K_in": 0.0},
        {"K_in": -1.0},
        {"Cl_out": 0.0},
        {"Cl_out": -1.0},
        {"Cl_in": 0.0},
        {"Cl_in": -1.0},
        {"Ca_out": 0.0},
        {"Ca_out": -1.0},
        {"Ca_in": 0.0},
        {"Ca_in": -1.0},
    ],
)
def test_non_positive_ion_concentration_raises(kwargs: dict):
    """Non-positive ion concentration must raise ValueError."""
    with pytest.raises(ValueError, match="concentration"):
        HodgkinHuxley(**kwargs)


def test_e_ca_matches_nernst(hh_model):
    """E_Ca must match Nernst equation with z=2 for default Ca concentrations."""
    expected = nernst_potential(2, hh_model.T, hh_model.Ca_out, hh_model.Ca_in)
    assert hh_model.E_Ca == pytest.approx(expected)


def test_e_ca_default_is_positive():
    """Default E_Ca should be large and positive (~131 mV)."""
    model = HodgkinHuxley()
    assert model.E_Ca > 100.0


def test_custom_ca_concentrations_shift_e_ca():
    """Changing Ca concentrations must produce a shifted E_Ca."""
    default_model = HodgkinHuxley()
    # Higher Ca_out → more positive E_Ca
    higher_out = HodgkinHuxley(Ca_out=10.0)
    assert higher_out.E_Ca > default_model.E_Ca
    # Lower Ca_out → less positive E_Ca
    lower_out = HodgkinHuxley(Ca_out=0.5)
    assert lower_out.E_Ca < default_model.E_Ca


# ---------------------------------------------------------------------------
# ion_concentrations tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "species, expected_out_attr, expected_in_attr",
    [
        (IonSpecies.SODIUM, "Na_out", "Na_in"),
        (IonSpecies.POTASSIUM, "K_out", "K_in"),
        (IonSpecies.CALCIUM, "Ca_out", "Ca_in"),
        (IonSpecies.CHLORIDE, "Cl_out", "Cl_in"),
    ],
)
def test_ion_concentrations_returns_correct_pair(
    hh_model: HodgkinHuxley,
    species: IonSpecies,
    expected_out_attr: str,
    expected_in_attr: str,
) -> None:
    """ion_concentrations returns the correct (C_out, C_in) pair for each species."""
    c_out, c_in = hh_model.ion_concentrations(species)
    assert c_out == pytest.approx(getattr(hh_model, expected_out_attr))
    assert c_in == pytest.approx(getattr(hh_model, expected_in_attr))


def test_ion_concentrations_reflects_custom_values() -> None:
    """ion_concentrations must return user-supplied concentration values."""
    model = HodgkinHuxley(Na_out=200.0, K_in=100.0, Ca_out=5.0, Cl_in=20.0)
    assert model.ion_concentrations(IonSpecies.SODIUM) == pytest.approx((200.0, 15.0))
    assert model.ion_concentrations(IonSpecies.POTASSIUM) == pytest.approx((5.0, 100.0))
    assert model.ion_concentrations(IonSpecies.CALCIUM) == pytest.approx((5.0, 0.0001))
    assert model.ion_concentrations(IonSpecies.CHLORIDE) == pytest.approx((120.0, 20.0))
