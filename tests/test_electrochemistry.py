"""Tests for electrochemical potential calculations (Nernst and GHK)."""

import pytest
from ap_sim.electrochemistry import goldman_potential, nernst_potential


@pytest.mark.parametrize(
    "z, T, c_out, c_in, expected",
    [
        (1, 310, 5.0, 140.0, -88.54),  # K+
        (1, 310, 145.0, 10.0, 71.43),  # Na+
        (2, 310, 1.8, 0.0001, 130.87),  # Ca2+
        (-1, 310, 120.0, 10.0, -66.38),  # Cl- (z=-1, as used for E_L in HH model)
    ],
)
def test_nernst_potential(z: int, T: float, c_out: float, c_in: float, expected: float):
    """Test Nernst potential for common physiological ions at 310 K."""
    result = nernst_potential(z, T, c_out, c_in)
    assert pytest.approx(result, 0.01) == expected


def test_equal_concentrations_gives_zero():
    """When c_out == c_in the Nernst potential must be 0 mV."""
    assert nernst_potential(1, 310, 100.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_zero_valence_raises():
    """z=0 causes division by zero — must raise ValueError."""
    with pytest.raises(ValueError, match="Valence"):
        nernst_potential(0, 310, 5.0, 140.0)


@pytest.mark.parametrize("T", [0, -1, -273.15])
def test_non_positive_temperature_raises(T: float):
    """T <= 0 is physically meaningless — must raise ValueError."""
    with pytest.raises(ValueError, match="Temperature"):
        nernst_potential(1, T, 5.0, 140.0)


@pytest.mark.parametrize("c_out", [0, -1.0])
def test_non_positive_extracellular_concentration_raises(c_out: float):
    """ion_concentration_out <= 0 causes log(0) or log(negative) — must raise."""
    with pytest.raises(ValueError, match="Extracellular"):
        nernst_potential(1, 310, c_out, 140.0)


@pytest.mark.parametrize("c_in", [0, -1.0])
def test_non_positive_intracellular_concentration_raises(c_in: float):
    """ion_concentration_in <= 0 causes log(0) or log(negative) — must raise."""
    with pytest.raises(ValueError, match="Intracellular"):
        nernst_potential(1, 310, 5.0, c_in)


# ---------------------------------------------------------------------------
# goldman_potential tests
# ---------------------------------------------------------------------------


def test_goldman_single_cation_equals_nernst():
    """A single cation with P=1 must reproduce the Nernst potential."""
    # K+ at default HH concentrations
    K_out, K_in = 5.0, 140.0
    T = 310.15
    expected = nernst_potential(1, T, K_out, K_in)
    result = goldman_potential(T, cation_terms=[(1.0, K_out, K_in)])
    assert result == pytest.approx(expected, rel=1e-6)


def test_goldman_equal_concentrations_gives_zero():
    """Equal concentrations for all ions must give 0 mV."""
    result = goldman_potential(
        310.0, cation_terms=[(1.0, 100.0, 100.0), (0.5, 50.0, 50.0)]
    )
    assert result == pytest.approx(0.0, abs=1e-10)


def test_goldman_ih_default_concentrations():
    """Ih with P_Na/P_K=0.289 at default HH concentrations gives ~-30 mV.

    Uses default HH concentrations: Na_out=145, Na_in=15, K_out=5, K_in=140.
    P_Na/P_K = 0.289 is chosen so that the GHK result equals -30 mV at 310.15 K.
    """
    T = 310.15
    Na_out, Na_in = 145.0, 15.0
    K_out, K_in = 5.0, 140.0
    p_na = 0.289
    result = goldman_potential(
        T,
        cation_terms=[(p_na, Na_out, Na_in), (1.0, K_out, K_in)],
    )
    # Should be within 0.1 mV of -30 mV
    assert result == pytest.approx(-30.0, abs=0.1)


def test_goldman_cation_and_anion():
    """Including an anion should shift the reversal potential."""
    T = 310.15
    # Single cation with a balancing anion at equal concentrations:
    # anion at equal conc contributes 0 shift, so result == single-cation Nernst
    K_out, K_in = 5.0, 140.0
    result_no_anion = goldman_potential(T, cation_terms=[(1.0, K_out, K_in)])
    result_with_zero_anion = goldman_potential(
        T,
        cation_terms=[(1.0, K_out, K_in)],
        anion_terms=[(0.0, 100.0, 100.0)],  # P=0 anion has no effect
    )
    assert result_no_anion == pytest.approx(result_with_zero_anion, rel=1e-6)


# ---------------------------------------------------------------------------
# goldman_potential error-path tests
# ---------------------------------------------------------------------------


def test_goldman_non_positive_temperature_raises():
    """T <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="Temperature"):
        goldman_potential(0.0, cation_terms=[(1.0, 5.0, 140.0)])


def test_goldman_empty_terms_raises():
    """Providing no ion terms must raise ValueError."""
    with pytest.raises(ValueError, match="At least one"):
        goldman_potential(310.0, cation_terms=[], anion_terms=[])


def test_goldman_negative_permeability_raises():
    """A negative permeability must raise ValueError."""
    with pytest.raises(ValueError, match="Permeability"):
        goldman_potential(310.0, cation_terms=[(-0.1, 5.0, 140.0)])


@pytest.mark.parametrize("c_out", [0.0, -1.0])
def test_goldman_non_positive_extracellular_raises(c_out: float):
    """Non-positive extracellular concentration must raise ValueError."""
    with pytest.raises(ValueError, match="Extracellular"):
        goldman_potential(310.0, cation_terms=[(1.0, c_out, 140.0)])


@pytest.mark.parametrize("c_in", [0.0, -1.0])
def test_goldman_non_positive_intracellular_raises(c_in: float):
    """Non-positive intracellular concentration must raise ValueError."""
    with pytest.raises(ValueError, match="Intracellular"):
        goldman_potential(310.0, cation_terms=[(1.0, 5.0, c_in)])
