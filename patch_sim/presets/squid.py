"""Squid giant axon (HH52) preset factory."""

from patch_sim.neuron import Neuron


def make_squid_giant_axon() -> Neuron:
    """Original Hodgkin-Huxley (1952) squid giant axon (HH52).

    Single-compartment model of the Loligo squid giant axon characterised
    at room temperature (~6 °C) in Hodgkin & Huxley (1952).  All channel
    parameters are at their HH52 defaults; no additional currents are added.

    Passive properties:
        area_cm2 is left as None because HH52 was characterised on a giant
        axon segment rather than a whole somatic cell.  Per-area density
        units (kΩ·cm², µF/cm²) remain the conventional display; absolute
        MΩ/pF values are not meaningful for this preparation.

    Temperature:
        Q10=1.0 — no thermal correction applied.  The published kinetics
        already represent the intact room-temperature preparation; scaling
        to mammalian temperature is not appropriate.

    Ion concentrations:
        K_out=7.8 mM restores the HH52 seawater value (E_K ≈ −77 mV).
        Default Na_out/Na_in concentrations produce E_Na ≈ +50 mV and
        E_L ≈ −54 mV, matching the original paper.

    References:
        - Hodgkin & Huxley (1952), J. Physiol. 117:500

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` using HH52 squid-axon
        parameters with seawater K_out (E_K ≈ −77 mV) and Q10=1.0.
    """
    return Neuron(
        K_out=7.8,
        Q10=1.0,
    )
