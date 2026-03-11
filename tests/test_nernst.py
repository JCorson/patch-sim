import pytest
from ap_sim.nernst import nernst_potential


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
