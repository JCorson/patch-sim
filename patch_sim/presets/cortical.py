"""Cortical RS pyramidal cell preset factory."""

from typing import Any

from patch_sim.channels import (
    make_ih_channel,
    make_im_channel,
    make_inap_channel,
    make_mainen_sejnowski_kv_channel,
    make_nav12_channel,
    make_pospischil_k_channel,
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
    # 0.3 µA/cm² peaks near −65 mV (strongly subthreshold) under the
    # Pospischil Na + M-S Kv pairing introduced in #311.
    SUBTHRESHOLD_RESPONSE: {
        "min_stimulus": 0.3,
        "max_stimulus": 0.3,
    },
    # 2 µA/cm² × 15 ms evokes a single AP under the Pospischil Na +
    # M-S Kv pairing (g_Na=35, Q10=1).  At 1.0 µA/cm² the new lower
    # g_Na fails to reach AP threshold within the 15 ms step.
    ACTION_POTENTIAL: {
        "min_stimulus": 2.0,
        "max_stimulus": 2.0,
        "stimulus_duration": 15.0,
    },
    # 800 ms at 5 µA/cm² is long enough for the (now reduced) IM to
    # accumulate and produce a measurable increase in inter-spike intervals
    # (spike-frequency adaptation); under g_M=0.075 the effect is modest
    # (~10–15% ISI growth) but reliably present.
    REPETITIVE_FIRING: {
        "min_stimulus": 5.0,
        "max_stimulus": 5.0,
        "stimulus_duration": 800.0,
        "pre_stimulus_duration": 50.0,
        "post_stimulus_duration": 50.0,
    },
    # Threshold is ~3–4 µA/cm²; 0 → 12 in steps of 1.5 (9 sweeps) spans
    # the subthreshold zone through fast repetitive firing.  300 ms is
    # long enough for IM-driven adaptation to be visible within each
    # suprathreshold sweep.
    FI_CURVE: {
        "max_stimulus": 12.0,
        "stimulus_step": 1.5,
        "stimulus_duration": 300.0,
    },
    # High R_in means small currents produce large deflections.  −5 → −1 µA/cm²
    # gives peaks of −109 to −80 mV with Ih-driven sag of 10–25 mV per step.
    # At the deepest steps the cell shows a post-hyperpolarization rebound
    # spike driven by two overlapping mechanisms: HH anode-break excitation
    # (h fully de-inactivates at −109 mV; n deactivates; m fires on release)
    # and Ih-driven overshoot (Ih activated during the step continues
    # conducting after release, transiently depolarising the membrane).  The
    # cell has no ICaT, so the rebound is not a low-threshold Ca²⁺ burst.
    HYPERPOLARIZATION_STEPS: {
        "min_stimulus": -5.0,
        "max_stimulus": -1.0,
        "stimulus_step": 1.0,
    },
}


def make_cortical_pyramidal() -> Neuron:
    """Cortical RS pyramidal cell (Pospischil 2008, Mainen-Sejnowski 1996).

    Single-compartment regular-spiking (RS) L5 pyramidal cell (~20 µm soma,
    area_cm2=20e-6 cm²).  The preset reproduces the broad AP (half-width
    1.0–2.5 ms), spike-frequency adaptation via IM/IKCa, and voltage sag from
    Ih that are characteristic of RS cortical pyramidal cells.

    Na/K kinetics and temperature:
        Pospischil et al. (2008) Traub-Miles Na⁺ kinetics (VT = −56.2 mV,
        via ``make_nav12_channel``) provide the fast spike upstroke.
        Mainen & Sejnowski (1996) Kv kinetics (``make_mainen_sejnowski_kv_channel``)
        serve as the sole active delayed rectifier; the Pospischil K factory is
        present but set to g_K=0.  The M-S Kv is essentially closed at rest
        (n_inf 50 % point near +4 mV) and deactivates on a τ ~1–2 ms scale —
        this slow-K balance is what broadens the AP into the 1.0–2.5 ms
        McCormick et al. (1985) half-width band.  A fast K (Pospischil n⁴)
        narrows the spike below 1 ms and is therefore excluded.
        Q10=1.0 with T_ref=307.15 K: Pospischil (2008) publish their kinetics
        at 34 °C without a Q10 correction; the M-S Kv rate constants are
        pre-scaled from 23→34 °C internally.  No further runtime scaling is
        applied.

    Slow Na inactivation and depolarisation-block recovery:
        Two complementary slow gates ensure the cell repolarises after
        sustained suprathreshold drive rather than hanging on a plateau:

        1. Fast-Na slow inactivation (sNa12, via ``make_nav12_channel``) —
           closes the residual fast-Na h-tail during prolonged depolarisation.
           Fleidervish & Gutnick (1996) characterised this gate directly in
           cortical pyramidal cells; g_Na=70 mS/cm² compensates for the ~8 %
           rest-state availability reduction (sNa12_inf ≈ 0.92 at −70 mV).
        2. INaP slow inactivation (sNaP, via ``make_inap_channel``) — removes
           the persistent Na⁺ window current that otherwise sustains the
           plateau (Magistretti & Alonso 1999).

        K_ATP is intentionally excluded: cortical pyramidal cells are not
        autonomous pacemakers, so the metabolic-safety K_ATP rescue is not
        biologically motivated.

    Additional subthreshold currents:
        Ih (g_h=0.3 mS/cm²) produces voltage sag and accelerates return from
        AHP; INaP (g_NaP=0.1 mS/cm²) provides a small persistent window
        current; IM (g_max=0.075 mS/cm²) drives spike-frequency adaptation.

    Passive properties:
        g_NaL + g_KL = 0.05 mS/cm² gives τ_m ≈ 20 ms and R_in ≈ 20 kΩ·cm²,
        reflecting the high input resistance (200–400 MΩ in vivo) of RS
        cortical pyramidal cells.  M-S Kv is essentially closed at rest, so
        the split is nearly pure K⁺ leak (g_KL ≈ 0.0496, g_NaL ≈ 0.0004).
        K_out=3.32 gives E_K ≈ −100 mV (Pospischil target).

    Known limitations:
        - ``find_zero_current_voltage`` with the default bracket [−100, −20]
          cannot find a unique zero: the slow M-S Kv contributes little outward
          current at −20 mV, so the bracket is not bounded by opposite-sign
          currents.  The cell rests stably at −70 mV in simulation, but the
          static-gating equilibrium analysis fails — see the
          ``test_find_zero_current_voltage_all_presets`` exclusion.

    References:
        - Pospischil et al. (2008), Biol. Cybern. 99:427 (Traub-Miles Na)
        - Mainen & Sejnowski (1996), Nature 382:363 (Kv kinetics)
        - McCormick et al. (1985), J. Neurophysiol. 54:782 (RS AP-shape bands)
        - Fleidervish & Gutnick (1996), J. Physiol. 493:83 (slow Na inactivation
          in cortical pyramidal cells — primary source)
        - Magistretti & Alonso (1999), J. Gen. Physiol. 114:491 (INaP slow
          inactivation)
        - Mickus, Jung & Spruston (1999), Biophys. J. 76:846 (slow Na
          inactivation, CA1 pyramidal — same Na-channel family)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~20 µm soma
        (area_cm2=20e-6 cm²) with Nav1.2/M-S Kv core, slow Na inactivation,
        Ih/INaP/IM/M-S Kv auxiliary channels, and a high-resistance leak.
    """
    return Neuron(
        g_Na=70.0,
        g_K=0.0,
        v_rest=-70.0,
        K_out=3.32,
        g_NaL=0.000391,
        g_KL=0.049609,
        Q10=1.0,
        T_ref=307.15,
        na_channel_factory=make_nav12_channel,
        k_channel_factory=make_pospischil_k_channel,
        additional_channels=(
            make_ih_channel(g_max=0.3),
            make_inap_channel(g_max=0.1),
            make_im_channel(g_max=0.075),
            make_mainen_sejnowski_kv_channel(g_max=1.8),
        ),
        area_cm2=20e-6,
    )
