"""Cerebellar Purkinje cell preset factory."""

from typing import Any

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ical_channel,
    make_icat_channel,
    make_ih_channel,
    make_ikca_channel,
    make_inap_channel,
    make_inar_channel,
    make_k_leak_channel,
    make_purkinje_k_channel,
    make_purkinje_na_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    FREQUENCY_RESPONSE,
    HYPERPOLARIZATION_STEPS,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    # Purkinje is a spontaneous pacemaker (fires without external current).
    # The Subthreshold Response and Action Potential protocols are kept for
    # UI display purposes; the cell will fire spontaneously during both.
    # Subthreshold Response: small positive current to demonstrate INaP
    # amplification of subthreshold depolarization just below threshold.
    SUBTHRESHOLD_RESPONSE: {
        "min_stimulus": 0.1,
        "max_stimulus": 0.1,
    },
    # 5 µA/cm² at 5 ms superimposed on spontaneous pacemaking.
    ACTION_POTENTIAL: {
        "min_stimulus": 5.0,
        "max_stimulus": 5.0,
        "stimulus_duration": 5.0,
    },
    # 10 µA/cm² for 180 ms drives sustained tonic firing with prominent
    # Ca²⁺ contributions (ICaL/ICaT/IKCa).  This is intrinsic tonic
    # firing, not the climbing-fiber-driven "complex spike" of in-vivo
    # Purkinje cells (which is unmodeled here).
    REPETITIVE_FIRING: {
        "min_stimulus": 10.0,
        "max_stimulus": 10.0,
        "stimulus_duration": 180.0,
    },
    # R_in ≈ 22.7 kΩ·cm²; −0.5 µA/cm² hyperpolarizes by ≈ 11 mV (to −76 mV)
    # and temporarily suppresses pacemaking, revealing Ih-driven sag and
    # rebound firing on step release.  −1.5 → −0.5 µA/cm² in 0.5 µA steps.
    HYPERPOLARIZATION_STEPS: {
        "min_stimulus": -1.5,
        "max_stimulus": -0.5,
        "stimulus_step": 0.5,
    },
    # Autonomous pacemaker: a hyperpolarizing holding current is required
    # to silence tonic firing during the chirp (mirrors a real ZAP
    # experiment under current-clamp hold).  −5 µA/cm² holds the cell at
    # ≈ −80 mV (below the INaP-driven oscillator window); amp=0.25 keeps
    # the response in the linear regime so the FFT-based impedance
    # analysis reports a clean profile.
    FREQUENCY_RESPONSE: {
        "dc_offset": -5.0,
        "amplitude": 0.25,
    },
}


def make_purkinje() -> Neuron:
    """Cerebellar Purkinje cell (De Schutter & Bower 1994, Carter & Bean 2009).

    Single-compartment somatic reduction (~25 µm soma with massive dendritic
    tree, area_cm2=250e-6 cm²; C ≈ 250 pF, matching Roth & Häusser 2001
    somatic recordings).  The preset reproduces autonomous tonic simple-spike
    pacemaking at ~10–50 Hz (in vivo rate 50–100 Hz requires dendritic inputs
    and network drive not modeled here).

    Pacemaking mechanism:
        INaP (Magistretti & Alonso 1999, V½ = −52.6 mV) provides a persistent
        Na⁺ window current that destabilizes any rest near −65 mV.  Ih
        (Destexhe et al. 1993, V½ ≈ −82 mV, g=1.0 mS/cm²) activates during
        the deep post-AP AHP and drives slow depolarization back to threshold,
        sustaining autonomous rhythmic firing.  v_rest=−65 mV is the simulation
        starting point; this cell has no stable zero-current resting potential.

    Na/K kinetics and temperature:
        De Schutter & Bower (1994) Traub-Miles Na⁺/K⁺ kinetics (VT = −58 mV)
        replace the HH52 defaults.  DSB94 channels were recorded at 32 °C;
        T_ref=305.15 K limits the runtime Q10 correction to ~1.73× (32→37 °C),
        preserving the published kinetics.  g_Na=30 mS/cm² (lower end of the
        somatic NaF range reported by Khaliq et al. 2003) and g_K=10 mS/cm²
        (a lumped delayed-rectifier representing multiple DSB94 K compartments)
        place the AP peak inside the +10 to +40 mV band and the AHP near
        −68 mV (Häusser & Clark 1997; Raman & Bean 1999).

    Slow Na inactivation and depolarization-block recovery:
        Two complementary slow gates ensure the cell repolarizes after
        sustained suprathreshold drive rather than hanging on a plateau:

        1. Fast-Na slow inactivation (sNa, via ``make_purkinje_na_channel``) —
           closes the residual fast-Na h-tail during prolonged depolarization.
           Carter & Bean (2009) directly demonstrated cumulative slow
           inactivation in cerebellar Purkinje somatic Na⁺ channels.
        2. INaP slow inactivation (sNaP, via ``make_inap_channel``) — removes
           the persistent Na⁺ window current that otherwise sustains the
           plateau (Magistretti & Alonso 1999).

        K_ATP is intentionally excluded: Carter & Bean's (2009) demonstration
        of depol-block recovery uses the two slow gates alone; K_ATP is not
        biologically motivated for Purkinje cells.

    Known limitations:
        Complex-spike "bursts" observed in vivo are climbing-fiber synaptic
        events (Raman & Bean 1997 — INaR-supported repriming under
        climbing-fiber EPSPs).  They are NOT intrinsic and cannot be produced
        by this single-compartment current-clamp preset.

    References:
        - De Schutter & Bower (1994), J. Neurophysiol. 71:375 (Na/K kinetics)
        - Carter & Bean (2009), Neuron 64:898 (slow Na inactivation in Purkinje)
        - Raman & Bean (1997), Neuron 19:881 (INaR; complex-spike mechanism)
        - Raman & Bean (1999), J. Neurosci. 19:4663 (INaP/INaR pacemaking)
        - Häusser & Clark (1997), J. Neurosci. 17:2358 (interspike V range)
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP slow
          inactivation)
        - Khaliq et al. (2003), J. Neurosci. 23:4899 (somatic NaF density)
        - Roth & Häusser (2001), J. Physiol. 535:445 (C and R_n)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~25 µm soma
        (area_cm2=250e-6 cm²) with DSB94 Na/K core, Carter-Bean slow Na
        inactivation, ICaL/ICaT/IKCa/INaP/INaR/Ih auxiliary channels, and
        tuned CalciumDynamics.
    """
    return Neuron(
        v_rest=-65.0,
        T_ref=305.15,
        channels=(
            make_purkinje_na_channel(g_max=30.0),
            make_purkinje_k_channel(g_max=10.0),
            # K-leak only — no Na-leak.
            make_k_leak_channel(g_max=0.044),
            make_ical_channel(g_max=1.0),
            make_icat_channel(g_max=0.5),
            make_ikca_channel(g_max=2.0),
            make_inap_channel(g_max=0.1),
            make_inar_channel(g_max=0.1),
            make_ih_channel(g_max=1.0),
        ),
        # tau_ca=20 ms prevents inter-spike Ca accumulation at 10 Hz pacemaking;
        # alpha_ca=1.5e-5 gives peak ca_i ~2.6 µM (≤ 5 µM) under REPETITIVE_FIRING.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.5e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=250e-6,
    )
