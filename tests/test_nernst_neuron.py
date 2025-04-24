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
