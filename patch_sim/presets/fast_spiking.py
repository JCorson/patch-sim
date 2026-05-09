"""Cortical fast-spiking interneuron preset factory."""

from typing import Any

from patch_sim.channels import (
    make_ikv31_channel,
    make_k_leak_channel,
    make_na_leak_channel,
    make_nav11_channel,
    make_pospischil_k_channel,
)
from patch_sim.constants import (
    ACTION_POTENTIAL,
    HYPERPOLARIZATION_STEPS,
    REPETITIVE_FIRING,
)
from patch_sim.neuron import Neuron

PROTOCOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    # High total leak (g_NaL+g_KL=1.5 mS/cm²) raises the firing threshold;
    # 30 µA/cm² is safely suprathreshold with the retuned Pospischil
    # kinetics (g_Na=80, g_K=30).  3–6 ms all produce exactly 1 AP;
    # 5 ms sits in the middle of that stable range.  ≥8 ms triggers a
    # second AP; the previous setting (25 µA/cm² · 5 ms) fails to reach
    # threshold under the retuned conductances (issue #301).
    ACTION_POTENTIAL: {
        "min_stimulus": 30.0,
        "max_stimulus": 30.0,
        "stimulus_duration": 5.0,
    },
    # 26 µA/cm² is just above the repetitive-firing threshold for the
    # retuned cell (≈68 spikes over 180 ms).  Lower amplitudes fire only
    # 1 AP; Pospischil kinetics sustain non-adapting high-frequency firing
    # without depolarization block.
    REPETITIVE_FIRING: {
        "min_stimulus": 26.0,
        "max_stimulus": 26.0,
        "stimulus_duration": 180.0,
    },
    # Very low R_in (~0.67 kΩ·cm²) requires large currents for noticeable
    # hyperpolarization.  −20 → −5 µA/cm² gives peaks of −74 to −67 mV.
    # No post-inhibitory rebound: Pospischil Na⁺/K⁺ kinetics at 34 °C do
    # not produce the Na⁺ de-inactivation overshoot seen in HH52 kinetics.
    HYPERPOLARIZATION_STEPS: {
        "min_stimulus": -20.0,
        "max_stimulus": -5.0,
        "stimulus_step": 5.0,
    },
}


def make_fast_spiking_interneuron() -> Neuron:
    """Cortical fast-spiking interneuron (Erisir 1999, Pospischil 2008).

    Single-compartment model of a cortical FS basket/chandelier interneuron
    (~10 µm soma, area_cm2=3e-6 cm²).  The preset reproduces non-adapting
    high-frequency firing (100–500 Hz) with a narrow AP (half-width ≈ 0.30 ms)
    and a shallow AHP, matching recorded FS phenotypes.

    Na/K kinetics:
        Pospischil et al. (2008) Traub-Miles kinetics replace the HH52
        defaults.  HH52 kinetics were characterised at room temperature and
        over-accelerate Na⁺ inactivation at mammalian temperature (Q10=3.0,
        22→37 °C, factor ~5.2×), causing depolarisation block after the first
        AP in a fast-spiking cell.  Pospischil kinetics were characterised at
        34 °C; T_ref=307.15 K limits the runtime Q10 correction to ~1.4×,
        preserving the published non-adapting phenotype.

    Slow Na inactivation (Nav1.1 isoform):
        ``make_nav11_channel`` adds a Nav1.1-flavoured slow inactivation gate
        (V½ = −45 mV, τ_floor = 5000 ms).  Nav1.1 is the dominant Na⁺ channel
        isoform in FS interneurons (Patel et al. 2015); the slow gate is too
        slow to accumulate significantly at FSI firing rates, so the overall
        FS phenotype is preserved.  At v_rest = −65 mV, sNa11_inf ≈ 0.92;
        g_Na=88 mS/cm² compensates for this small rest-availability reduction.

    IKv3.1 delayed rectifier:
        Kv3.1-type K⁺ channel (V₁/₂ = −12.4 mV, fast deactivation) drives the
        late-AP repolarisation and enables non-adapting high-frequency firing
        (Erisir et al. 1999; Wang & Buzsáki 1996).  The Pospischil delayed
        rectifier handles early repolarisation; IKv3.1 sharpens the trailing
        edge and prevents inter-spike Na⁺ channel inactivation accumulation.

    Passive properties:
        g_NaL + g_KL = 1.5 mS/cm² gives τ_m ≈ 0.67 ms — a highly leaky
        membrane that narrows the synaptic integration window, a hallmark of
        FS interneurons (Erisir et al. 1999).  R_n ≈ 220 MΩ and C ≈ 3 pF
        match reported FS interneuron values.

    AP-shape acceptance bands (Erisir et al. 1999; Kawaguchi 1995):
        - AP peak: ~+38 mV  (band +10 to +40 mV)
        - AHP depth: ~−74 mV  (band −78 to −60 mV)
        - Half-width: ~0.30 ms  (band 0.25–0.7 ms)
        - Firing rate at 30 µA/cm²: ~235 Hz  (band 100–500 Hz)

    References:
        - Erisir et al. (1999), J. Neurophysiol. 82:2476 (Kv3.1 kinetics, FS bands)
        - Pospischil et al. (2008), Biol. Cybern. 99:427 (Traub-Miles Na/K)
        - Kawaguchi (1995), J. Neurosci. 15:2638 (FS phenotype)
        - Wang & Buzsáki (1996), J. Neurosci. 16:6402 (IKv3.1 role)
        - Patel et al. (2015), Nat. Commun. 6:8249 (Nav1.1 slow inactivation)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~10 µm soma
        (area_cm2=3e-6 cm²) with Pospischil Na/K core, Nav1.1 slow
        inactivation, IKv3.1 auxiliary channel, and a high-leak membrane.
    """
    return Neuron(
        T_ref=307.15,
        channels=(
            make_nav11_channel(g_max=88.0),
            make_pospischil_k_channel(g_max=30.0),
            make_na_leak_channel(g_max=0.3115),
            make_k_leak_channel(g_max=1.1885),
            make_ikv31_channel(g_max=20.0),
        ),
        area_cm2=3e-6,
    )
