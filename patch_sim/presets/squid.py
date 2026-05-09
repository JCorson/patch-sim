"""Squid giant axon (HH52) preset factory."""

from typing import Any

from patch_sim.channels import (
    make_k_channel,
    make_k_leak_channel,
    make_na_channel,
    make_na_leak_channel,
)
from patch_sim.constants import ACTION_POTENTIAL
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    # 10 µA/cm² at 30 ms (default) produces 2 spikes; the membrane
    # recovers and re-fires within the step.  10 ms is long enough to
    # reach threshold and evoke exactly one action potential.
    ACTION_POTENTIAL: {
        "stimulus_duration": 10.0,
    },
    # Squid inherits the base preset range (−10 → −2 µA/cm²), which
    # drives the membrane to roughly −98 mV.  At that depth the Na⁺
    # inactivation gate h fully de-inactivates (h_inf ≈ 0.996, τ_h ≈
    # 2.5 ms — reached within ~10 ms) and the K⁺ activation gate n
    # deactivates (n_inf ≈ 0.002).  On step release, m activates rapidly
    # while h is still elevated and g_K is negligible, triggering a
    # post-hyperpolarization action potential — classic HH anode-break
    # excitation (Hodgkin & Huxley, 1952).  No ICaT or Ih channels are
    # involved; the spike is an intrinsic property of the HH52 Na/K model
    # at these depths.
}


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
        channels=(
            make_na_channel(g_max=120.0),
            make_k_channel(g_max=36.0),
            make_na_leak_channel(g_max=0.054),
            make_k_leak_channel(g_max=0.246),
        ),
    )
