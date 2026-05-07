"""Cortical fast-spiking interneuron preset factory."""

from patch_sim.channels import (
    make_ikv31_channel,
    make_nav11_channel,
    make_pospischil_k_channel,
)
from patch_sim.neuron import Neuron


def make_fast_spiking_interneuron() -> Neuron:
    """Cortical fast-spiking interneuron (Erisir 1999, Pospischil 2008).

    Returns:
        Neuron configured with Pospischil Na/K kinetics, Nav1.1 slow
        inactivation, and IKv3.1 to drive non-adapting high-frequency firing.
    """
    # area_cm2 = 3e-6 cm² — compact ~10 µm soma, minimal dendrites.
    # With g_total ≈ 1.5 mS/cm², gives R_n ≈ 220 MΩ and C ≈ 3 pF
    # (matching reported FS interneuron values; Erisir et al. 1999).
    #
    # Active conductances (g_Na=80, g_K=30, g_IKv3.1=20) deliver the
    # fast-spiking phenotype while keeping the AP peak and AHP depth inside
    # the bands reported by Erisir et al. (1999) and Kawaguchi (1995):
    #   AP peak  ~+39 mV  (band +10 to +40 mV)
    #   AHP depth ~−74 mV (band −78 to −60 mV)
    # The earlier g_Na=150, g_K=50, g_IKv3.1=40 setting overshot to peak
    # ~+44 mV and AHP ~−81 mV — both Na and K drive were larger than
    # needed, pulling V toward E_Na on the upstroke and E_K on the
    # undershoot.  Halving each conductance preserves the high firing rate
    # (~237 Hz at 30 µA/cm², well inside the 100–500 Hz FS band) and keeps
    # half-width ~0.30 ms (band 0.25–0.7 ms), while bringing peak and AHP
    # into band (issue #301).
    #
    # IKv3.1 (Kv3.1-type, V₁/₂=−12.4 mV, fast deactivation) drives the
    # late-AP repolarization phase and supports non-adapting firing; the
    # Pospischil delayed rectifier handles the bulk of the early
    # repolarization.  Refs: Erisir et al. (1999), J. Neurophysiol. 82:2476;
    # Kawaguchi (1995), J. Neurosci. 15:2638; Wang & Buzsáki (1996),
    # J. Neurosci. 16:6402.
    #
    # Pospischil et al. (2008) Traub-Miles Na⁺/K⁺ kinetics replace the
    # default HH52 core channels.  HH52 kinetics (fitted to room-temperature
    # squid axon) over-accelerate Na⁺ inactivation under the default
    # Q10=3.0 scaling (22→37 °C, factor ~5.2×), causing depolarization
    # block after the first AP — biologically wrong for a fast-spiking cell.
    # Pospischil kinetics were characterized at mammalian temperature (34 °C),
    # so T_ref=307.15 K limits the Q10 factor to ~1.4× and the cell sustains
    # non-adapting high-frequency firing as expected.
    #
    # Fast Na⁺ uses ``make_nav11_channel`` (Pospischil base + a weak
    # Nav1.1-flavoured slow inactivation gate, V½ = −45 mV, τ_floor =
    # 5000 ms) so the model captures the biological fact that Nav1.1
    # has slow inactivation (Patel et al. 2015) while the kinetics
    # remain too slow to dominate at FSI firing rates.  At v_rest =
    # −65 mV the gate sits at sNa11_inf ≈ 0.92, so g_Na is bumped
    # from the previous 80 → 88 mS/cm² to compensate for the ~8 %
    # rest-availability reduction; AP peak (~+38 mV), AHP (~−74 mV),
    # half-width (~0.30 ms), and ~235 Hz firing at 30 µA/cm² stay
    # within the Erisir / Kawaguchi / Wang-Buzsáki bands.  A higher-
    # fidelity isoform-fitted overhaul (true Nav1.1 activation/fast-
    # inactivation kinetics from Hu & Jonas 2014) is tracked as a
    # follow-up.
    #
    # g_NaL + g_KL = 1.5 mS/cm² gives τ_m ≈ 0.67 ms — highly leaky membrane
    # that narrows the synaptic integration window, a hallmark of FS cells.
    # Values originally tuned so that I_NaL + I_KL + I_channels = 0 at
    # v_rest = −65 mV with K_out=4 mM (E_K ≈ −95 mV) and the previous
    # higher-conductance setting.  At rest the Pospischil m³h ≈ 1.1e-8 and
    # n⁴ ≈ 4.4e-9, and IKv3.1 nk² ≈ 1.3e-4 — small enough that halving
    # each active conductance does not perturb v_rest measurably; the leak
    # split is therefore kept identical, and `test_fs_no_spontaneous_firing`
    # confirms the cell still rests stably below threshold.
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
