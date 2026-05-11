"""Thalamic reticular neuron preset factory."""

from typing import Any

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    make_ih_channel,
    make_ikca_channel,
    make_k_leak_channel,
    make_na_leak_channel,
    make_trn_icat_channel,
    make_trn_k_channel,
    make_trn_na_channel,
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
    # Very low threshold with HP92 kinetics; 0.01 µA/cm² is safely subthreshold.
    SUBTHRESHOLD_RESPONSE: {
        "min_stimulus": 0.01,
        "max_stimulus": 0.01,
    },
    # TRN is a tonic pacemaker post-#308 retune (~10 Hz spontaneous from
    # v_rest = −80 mV with ICaT half-deinactivated), so within the 22 ms
    # ACTION_POTENTIAL window the cell fires several APs regardless of
    # injected current — there is no "single-AP" pulse for this preset.
    # Excluded from ``test_action_potential_preset`` via
    # ``_QUIESCENT_PRESET_NAMES`` because the strict "exactly 1 AP"
    # assertion does not apply to a pacemaker.  Stimulus parameters are
    # kept at the pre-#308 values (5 µA/cm² × 2 ms) so users still see a
    # stim-evoked perturbation on top of the spontaneous tonic train.
    ACTION_POTENTIAL: {
        "min_stimulus": 5.0,
        "max_stimulus": 5.0,
        "stimulus_duration": 2.0,
    },
    # Depolarizing step for sustained repetitive firing via ICaT;
    # 3 µA/cm² gives ≥5 spikes over 200 ms with HP92 kinetics.
    REPETITIVE_FIRING: {
        "min_stimulus": 3.0,
        "max_stimulus": 3.0,
        "stimulus_duration": 200.0,
    },
    # Low threshold with HP92 kinetics; 0–5 µA/cm² in 0.5 µA steps.
    FI_CURVE: {
        "max_stimulus": 5.0,
        "stimulus_step": 0.5,
        "stimulus_duration": 100.0,
    },
    # Sweep −5 → −1 µA/cm² (in 1.0 µA steps) for 500 ms with 100 ms pre
    # and 300 ms post.  At ≤ −2 µA/cm² the cell de-inactivates ICaT and
    # activates Ih enough to fire the HP92 rebound burst on release;
    # the −1 µA step is included as a reference subthreshold sweep so
    # the burst is visible by contrast.  Step duration is 500 ms (vs
    # the 300 ms default) so Ih has time to fully activate.
    HYPERPOLARIZATION_STEPS: {
        "pre_stimulus_duration": 100.0,
        "stimulus_duration": 500.0,
        "post_stimulus_duration": 300.0,
        "min_stimulus": -5.0,
        "max_stimulus": -1.0,
        "stimulus_step": 1.0,
    },
}


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
        hyperpolarization.  Ih (g=0.020 mS/cm²; Bal & McCormick 1993 report
        ≈0.025 for cat TRN) activates during the hyperpolarizing step and
        drives V across the LTS threshold on release — without Ih, passive
        relaxation does not overshoot the threshold and the LTS does not fire.
        IKCa (HP92, g=0.3 mS/cm²) converts ICaT-mediated Ca²⁺ entry into
        outward K⁺ current, generating the AHP after spikes during tonic firing
        and contributing to burst termination.

    ICaT factory and inactivation time constant:
        ``make_trn_icat_channel`` replaces the default cosh-shaped Destexhe
        (1994) ft tau with a sigmoid-shaped tau — small (~20 ms) at
        hyperpolarized V and large (~200 ms) at LTS-plateau V.  This sustains
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
        half-inactivation point (ft_inf ≈ 0.50), favoring post-inhibitory
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
        T_ref=309.15,
        channels=(
            make_trn_na_channel(g_max=50.0),
            make_trn_k_channel(g_max=24.0),
            make_na_leak_channel(g_max=0.0066),
            make_k_leak_channel(g_max=0.0634),
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
