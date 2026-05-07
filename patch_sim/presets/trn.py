"""Thalamic reticular neuron preset factory."""

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ih_channel,
    make_ikca_channel,
    make_trn_icat_channel,
    make_trn_k_channel,
    make_trn_na_channel,
)
from patch_sim.neuron import Neuron


def make_trn() -> Neuron:
    """Thalamic reticular neuron (Huguenard & Prince 1992, Bal & McCormick 1993).

    Single-compartment model (~15 µm soma, area_cm2=7e-6 cm²; C ≈ 7 pF).
    The preset reproduces the HP92 LTS-rebound multi-spike burst phenotype
    (5–15 Na⁺ spikes at 200–600 Hz) and autonomous tonic pacemaking at ~10 Hz
    (HP92 and Bal & McCormick 1993 report 1–15 Hz in slice).  v_rest=−80 mV
    is the configured initial condition; the cell rapidly leaves this point and
    settles into tonic firing.

    LTS burst mechanism:
        ICaT (``make_trn_icat_channel``, g=2.85 mS/cm²) produces a
        low-threshold Ca²⁺ spike plateau when ft is de-inactivated by prior
        hyperpolarisation.  Ih (g=0.020 mS/cm²; Bal & McCormick 1993 report
        ≈0.025 for cat TRN) activates during the hyperpolarising step and
        drives V across the LTS threshold on release — without Ih, passive
        relaxation does not overshoot the threshold and the LTS does not fire.
        IKCa (HP92, g=0.3 mS/cm²) converts ICaT-mediated Ca²⁺ entry into
        outward K⁺ current, generating the AHP after spikes during tonic firing
        and contributing to burst termination.

    ICaT factory and inactivation time constant:
        ``make_trn_icat_channel`` replaces the default cosh-shaped Destexhe
        (1994) ft tau with a sigmoid-shaped tau — small (~20 ms) at
        hyperpolarised V and large (~200 ms) at LTS-plateau V.  This sustains
        the LTS plateau long enough to support the 5–15 spike HP92 burst.
        ft_inf(V) is unchanged from Destexhe (1994).

    Na/K kinetics and temperature:
        Huguenard & Prince (1992) / Pospischil (2008) Traub-Miles Na⁺/K⁺
        kinetics (VT = −67 mV) replace the HH52 defaults.  HP92 channels were
        recorded at 36 °C; T_ref=309.15 K limits the runtime Q10 correction to
        ~1.12× (36→37 °C), preserving the published kinetics.  g_Na=50 mS/cm²
        and g_K=24 mS/cm² are explicit somatic densities that land every HP92
        tonic AP-shape metric (peak +10 to +40 mV; AHP −75 to −55 mV;
        half-width; threshold; firing rate) inside its band while preserving
        the 5–15 spike LTS burst at g_T=2.85 mS/cm².

    Passive properties:
        g_NaL=0.0066 / g_KL=0.0634 mS/cm² (total 0.07) gives τ_m ≈ 14.3 ms
        and R_in ≈ 14.3 kΩ·cm².  At v_rest=−80 mV the ft gate is near its
        half-inactivation point (ft_inf ≈ 0.50), favouring post-inhibitory
        rebound bursting.

    Known limitations:
        - Excluded from ``test_all_presets_stable_at_rest`` — autonomous
          oscillator with no stable zero-current equilibrium.

    References:
        - Huguenard & Prince (1992), J. Neurosci. 12:3804 (TRN biophysics;
          ICaT, IKCa; published g_KCa)
        - Bal & McCormick (1993), J. Physiol. 468:669 (Ih in cat TRN)
        - Destexhe et al. (1994), J. Neurophysiol. 72:803 (ICaT kinetics)
        - Pospischil et al. (2008), Biol. Cybern. 99:427, Table 2 (RE params)

    Returns:
        Fully-configured :class:`~patch_sim.Neuron` — ~15 µm soma
        (area_cm2=7e-6 cm²) with HP92/Pospischil Na/K core, TRN-tuned ICaT,
        IKCa, Ih, and tuned CalciumDynamics.
    """
    return Neuron(
        v_rest=-80.0,
        g_Na=50.0,
        g_K=24.0,
        g_NaL=0.0066,
        g_KL=0.0634,
        T_ref=309.15,
        na_channel_factory=make_trn_na_channel,
        k_channel_factory=make_trn_k_channel,
        additional_channels=(
            make_trn_icat_channel(g_max=2.85),
            make_ikca_channel(g_max=0.3),
            make_ih_channel(g_max=0.020),
        ),
        # Peak ca_i reaches ~9–10 µM under REPETITIVE_FIRING and 8–18 µM under
        # HYPERPOLARIZATION_STEPS (LTS rebound burst) — physiologically
        # consistent with TRN somatic Ca during high-frequency burst trains
        # (Cueni et al. 2008, Nat. Neurosci. 11:683).  The elevated Ca is
        # load-bearing: lower alpha_ca collapses the burst phenotype because
        # IKCa (g_KCa=0.3 mS/cm²) cannot terminate the burst cleanly without
        # sufficient Ca²⁺ drive.  ``test_calcium_calibration.py`` uses a
        # TRN-specific 12 µM upper bound to accommodate this realistic Ca level.
        calcium_dynamics=CalciumDynamics(alpha_ca=1.2e-5, tau_ca=20.0, ca_rest=1e-4),
        area_cm2=7e-6,
    )
