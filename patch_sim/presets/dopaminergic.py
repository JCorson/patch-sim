"""Substantia nigra dopaminergic neuron preset factory."""

from typing import Any

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_cav13_channel,
    make_dopaminergic_k_channel,
    make_dopaminergic_na_channel,
    make_ih_channel,
    make_k_leak_channel,
    make_na_leak_channel,
    make_sk_channel,
    make_snc_inap_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    FI_CURVE,
    HYPERPOLARIZATION_STEPS,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    # Subthreshold: Canavier/Komendantov kinetics (VT=-67 mV) lower the
    # firing threshold to ~0.3 µA/cm² for a 30 ms step; 0.1 µA/cm² is
    # comfortably sub-threshold and produces a passive depolarization.
    SUBTHRESHOLD_RESPONSE: {
        "min_stimulus": 0.1,
        "max_stimulus": 0.1,
    },
    # 4 µA/cm² at 5 ms evokes a single AP; lower amplitudes are subthreshold
    # and higher amplitudes (≥12 µA/cm²) fire two APs within 30 ms.
    ACTION_POTENTIAL: {
        "min_stimulus": 4.0,
        "max_stimulus": 4.0,
        "stimulus_duration": 5.0,
    },
    # SNc DA pacemaker: tonic firing throughout, accelerating modestly
    # with depolarizing drive.  The somatic single-compartment model
    # does not reproduce depolarization block at any tested amplitude
    # × duration (empirical sweep in scratch/characterize_da_block.py
    # — tonic firing up to 15 µA/cm² × 5 s and 2 µA/cm² × 10 s).
    # Real SNc DA neurons enter block above ~100 pA sustained drive
    # (Tucker et al. 2012); reproducing this requires dendritic Na
    # inactivation absent from this representation (#323).
    # The REPETITIVE_FIRING protocol uses 0.3 µA/cm² × 3000 ms,
    # producing ≥30 full APs at ~10 Hz over 3 s.  Duration must stay
    # > 180 ms (the base REPETITIVE_FIRING preset) — see
    # test_neuron_protocol_adjustments_change_stimulus_duration.
    REPETITIVE_FIRING: {
        "min_stimulus": 0.3,
        "max_stimulus": 0.3,
        "stimulus_duration": 3000.0,
    },
    # Threshold ~1 µA/cm²; 0 → 12 µA/cm² in 1.5 µA steps spans the
    # subthreshold zone through repetitive firing.  200 ms duration shows
    # the steady-state F-I relationship.
    FI_CURVE: {
        "max_stimulus": 12.0,
        "stimulus_step": 1.5,
        "stimulus_duration": 200.0,
    },
    # R_in ≈ 3.3 kΩ·cm²; −20 → −5 µA/cm² gives peaks of −106 to −71 mV
    # with clear Ih-driven sag (25–8 mV).  At step release, Ih (g=2.0 mS/cm²
    # activated during the step) drives a transient depolarization above
    # threshold, producing a rebound spike at the most-negative step.  This is
    # an Ih-mediated rebound using Canavier/Komendantov Na⁺ kinetics (VT = −67 mV
    # gives m_inf ≈ 27% at −48 mV, enough for Ih to trigger firing).  The cell
    # has no ICaT.
    HYPERPOLARIZATION_STEPS: {
        "min_stimulus": -20.0,
        "max_stimulus": -5.0,
        "stimulus_step": 5.0,
    },
}


def make_dopaminergic() -> Neuron:
    """Substantia nigra pars compacta dopaminergic neuron (Putzier+Drion).

    Single-compartment somatic model (~20 µm soma, area_cm2=50e-6 cm²;
    C ≈ 50 pF, R_in ≈ 500 MΩ, τ_m ≈ 25 ms — matching Wolfart et al. 2001
    and Lacey et al. 1989).  The preset reproduces autonomous tonic pacemaking
    at ~7 Hz, within the in-vitro SNc DA band of 1–8 Hz.

    Pacemaking mechanism:
        Drion et al. (2011) showed that Cav1.3 and INaP_SNc are
        interchangeable negative-slope conductances; both are required for a
        faithful model.  Pacemaking proceeds as:

        1. INaP_SNc (V½ = −65 mV) initiates the subthreshold depolarizing
           ramp from the AHP; Cav1.3 (V½ = −31.1 mV) takes over near threshold
           and loads Ca²⁺ into the cell.
        2. An AP fires; ca_i peaks at ~0.6 µM (α_ca = 5e-5 mM·cm²/µA).
        3. SK (K_d = 0.3 µM, Hill n=4, g=1.75 mS/cm²) opens rapidly, driving
           a clean medium AHP at ~−90 mV within <5 ms.  Without strong SK
           activation the Komendantov Na window current (m_inf=0.79 at −30 mV)
           holds V on a ~20 ms plateau — SK is essential to terminate each AP
           cleanly.
        4. ca_i decays (τ_ca = 30 ms), SK closes, INaP_SNc + Cav1.3 ramp
           resumes.  Cycle repeats at ~7 Hz.

    Na/K kinetics and temperature:
        Canavier (1999) / Komendantov (2004) kinetics (VT = −67 mV) replace
        the HH52 defaults.  Q10=1.0 with T_ref=308.15 K (35 °C) holds the
        rate constants at the published Komendantov reference temperature.
        Repolarization is carried by the Komendantov Kdr alone; IM and M-S Kv
        are excluded as cortical-specific channels inappropriate for SNc DA.

    Slow Na inactivation:
        ``make_dopaminergic_na_channel`` carries a cumulative slow Na gate
        (sNa_da; Khaliq & Bean 2010; Tucker et al. 2012) and
        ``make_snc_inap_channel`` carries a slow INaP gate (sNaP_snc;
        Magistretti & Alonso 1999, V½ tuned to the Drion 2011 SNc fit).
        These gates improve biological accuracy and would provide an escape
        from depol-block in a more complete model.

    Known limitations:
        Real SNc DA neurons enter depolarization block above ~100 pA sustained
        drive (Tucker et al. 2012).  This single-compartment somatic model does
        not reproduce block onset: empirical sweeps confirm tonic firing at all
        tested amplitudes.  Block onset requires the dendritic Na inactivation
        pool absent from this somatic reduction.  Tracked in issue #323.

    References:
        - Putzier, Kullmann & Roeper (2009), J. Neurosci. 29:15414 (Cav1.3
          drives SNc pacemaking)
        - Drion, Massotte, Sepulchre & Seutin (2011), PLOS Comput. Biol.
          7:e1002050 (Cav1.3 + INaP both required)
        - Komendantov et al. (2004), J. Neurophysiol. 91:346 (Na/K kinetics)
        - Wolfart, Neuhoff, Franz & Roeper (2001), J. Neurosci. 21:3443 (SK)
        - Tucker, Huertas, Horn et al. (2012), J. Neurophysiol. 108:288
          (depol-block onset ~100 pA)
        - Khaliq & Bean (2010), J. Neurosci. 30:7558 (slow Na inactivation,
          SNc DA)
        - Grace & Bunney (1984), J. Neurosci. 4:2877 (1–5 Hz in vitro)
        - Lacey, Mercuri & North (1989), J. Physiol. 415:55 (interspike V
          trough −55 to −65 mV)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~20 µm soma
        (area_cm2=50e-6 cm²) with Komendantov Na/K core, slow Na inactivation,
        Cav1.3/SK/INaP_SNc/Ih auxiliary channels, and tuned CalciumDynamics.
    """
    return Neuron(
        v_rest=-55.0,
        Q10=1.0,
        T_ref=308.15,
        channels=(
            make_dopaminergic_na_channel(g_max=10.0),
            make_dopaminergic_k_channel(g_max=0.5),
            make_na_leak_channel(g_max=0.012),
            make_k_leak_channel(g_max=0.028),
            make_cav13_channel(g_max=0.04),
            make_sk_channel(g_max=1.75),
            make_snc_inap_channel(g_max=0.012),
            make_ih_channel(g_max=0.20),
        ),
        calcium_dynamics=CalciumDynamics(
            alpha_ca=5.0e-5, tau_ca=30.0, ca_rest=1e-4, ca_init=1.0e-4
        ),
        area_cm2=50e-6,
    )
