"""Cortical fast-spiking interneuron preset factory."""

from patch_sim.channels import (
    make_ikv31_channel,
    make_nav11_channel,
    make_pospischil_k_channel,
)
from patch_sim.neuron import Neuron


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
        g_Na=88.0,
        g_K=30.0,
        g_NaL=0.3115,
        g_KL=1.1885,
        T_ref=307.15,
        na_channel_factory=make_nav11_channel,
        k_channel_factory=make_pospischil_k_channel,
        additional_channels=(make_ikv31_channel(g_max=20.0),),
        area_cm2=3e-6,
    )
