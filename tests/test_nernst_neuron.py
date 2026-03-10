import pytest
from ap_sim.nernst_neuron import nernst_potential


def test_potassium():
    """Test Nernst potential for potassium ion at 310 K."""
    z = 1  # Valence of K+
    T = 310  # Temperature in Kelvin
    ion_concentration_out = 5.0  # Extracellular concentration in mM
    ion_concentration_in = 140.0  # Intracellular concentration in mM
    expected = -88.54  # Expected result in mV (approximate)
    result = nernst_potential(z, T, ion_concentration_out, ion_concentration_in)
    assert pytest.approx(result, 0.01) == expected


def test_sodium():
    """Test Nernst potential for sodium ion at 310 K."""
    z = 1  # Valence of Na+
    T = 310  # Temperature in Kelvin
    ion_concentration_out = 145.0  # Extracellular concentration in mM
    ion_concentration_in = 10.0  # Intracellular concentration in mM
    expected = 71.43  # Expected result in mV (approximate)
    result = nernst_potential(z, T, ion_concentration_out, ion_concentration_in)
    assert pytest.approx(result, 0.01) == expected


def test_calcium():
    """Test Nernst potential for calcium ion at 310 K."""
    z = 2  # Valence of Ca2+
    T = 310  # Temperature in Kelvin
    ion_concentration_out = 1.8  # Extracellular concentration in mM
    ion_concentration_in = 0.0001  # Intracellular concentration in mM
    expected = 130.87  # Expected result in mV (approximate)
    result = nernst_potential(z, T, ion_concentration_out, ion_concentration_in)
    assert pytest.approx(result, 0.01) == expected


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
