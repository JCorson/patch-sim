"""Squid giant axon (HH52) preset factory."""

from patch_sim.neuron import Neuron


def make_squid_giant_axon() -> Neuron:
    """Original Hodgkin-Huxley (1952) squid giant axon (HH52).

    Returns:
        Neuron configured with HH52 squid-axon parameters and seawater K_out
        (E_K ≈ −77 mV).  Q10=1.0 (no thermal correction — this preset *is* the
        room-temperature squid axon model).
    """
    # area_cm2 left as None: HH52 was characterised on a giant axon segment
    # rather than a whole somatic cell, so absolute MΩ/pF values are not
    # the conventional way to report squid passive properties.  Per-area
    # density units (kΩ·cm², µF/cm²) remain the meaningful display.
    #
    # Original Hodgkin-Huxley (1952) parameters — all defaults.
    # Default ion concentrations produce HH52 reversal potentials
    # (E_Na ≈ +50, E_K ≈ −77, E_L ≈ −54 mV) so no overrides needed.
    # Ref: Hodgkin & Huxley (1952), J. Physiol. 117:500
    #
    # K_out=7.8 mM overrides DEFAULT_K_OUT (4.0 mM, mammalian ACSF) to
    # restore the HH52 seawater value (E_K ≈ −77 mV).
    #
    # Q10=1.0: this preset IS the room-temperature squid axon model.
    # Applying a 5.2× thermal correction to bring it to mammalian
    # temperature is not meaningful — the kinetics are already those
    # of the intact preparation.
    return Neuron(
        K_out=7.8,
        Q10=1.0,
    )
