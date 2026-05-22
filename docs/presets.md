# Neuron presets

A **preset** is a zero-argument factory that returns a fully configured
[`Neuron`](api/neuron.md) — with the right ion channels, conductances, ion concentrations,
temperature, and (where relevant) calcium dynamics for a particular cell type.
Presets are the fastest way to get biologically realistic behavior without
hand-assembling channels.

## Using presets

The presets are registered in two parallel structures:

```python
import patch_sim

# Map of display name -> factory function.
patch_sim.NEURON_PRESETS            # {"Squid Giant Axon (Classic HH)": <fn>, ...}
patch_sim.NEURON_PRESET_NAMES       # ordered list of the display names

# Build a neuron from a preset:
neuron = patch_sim.NEURON_PRESETS["Cortical Pyramidal Neuron"]()
```

Each cell type also defines **protocol adjustments** — per-cell tweaks to the
default stimulus parameters (amplitude, duration, step range) so that each
protocol lands in the physiologically interesting range for that neuron. They
are exposed as `patch_sim.NEURON_PROTOCOL_ADJUSTMENTS` and applied automatically
by `build_protocol_from_preset` (see [Protocols & analysis](protocols-and-analysis.md)).

The factory functions are also importable directly: `make_squid_giant_axon`,
`make_fast_spiking_interneuron`, `make_cortical_pyramidal`, `make_purkinje`,
`make_dopaminergic`, `make_thalamic_relay`, `make_ca1_pyramidal`, `make_stn`,
`make_trn`.

## The nine presets

### Squid Giant Axon (Classic HH)

The original Hodgkin & Huxley (1952) model of the *Loligo* squid giant axon.
Carries only the classic fast sodium, delayed-rectifier potassium, and sodium/
potassium leak channels. Characterized at room temperature, so `Q10 = 1.0` (no
thermal correction), with seawater `K_out = 7.8 mM` (E_K around -77 mV). Produces
textbook single and repetitive action potentials, and the classic anode-break
(post-hyperpolarization) rebound spike.

### Fast-Spiking Interneuron

A cortical fast-spiking basket/chandelier interneuron (Erisir 1999, Pospischil
2008). Uses Pospischil Nav1.1 sodium, a Pospischil delayed-rectifier potassium,
and a Kv3.1 channel for rapid repolarization, plus leaks. Fires non-adapting,
high-frequency trains (roughly 100-500 Hz) with a narrow action potential
(half-width around 0.30 ms) and a shallow afterhyperpolarization.

### Cortical Pyramidal Neuron

A neocortical regular-spiking pyramidal cell. Combines Pospischil Nav1.2 sodium
with a Mainen-Sejnowski Kv potassium channel, plus the hyperpolarization-
activated current Ih, the muscarinic potassium current IM, and a persistent
sodium current INaP. The IM current drives spike-frequency adaptation during
sustained firing; Ih produces sag and rebound on hyperpolarizing steps.

### Purkinje Neuron

A cerebellar Purkinje cell. A rich channel set — Purkinje-tuned sodium and
potassium, a resurgent sodium current INaR, persistent sodium INaP, L- and
T-type calcium (ICaL, ICaT), calcium-activated potassium IKCa, and Ih — coupled
to intracellular calcium dynamics. Fires as a spontaneous tonic pacemaker (it
spikes with no injected current). The in-vivo climbing-fiber "complex spike" is
not modeled.

### SNc Dopaminergic Neuron

A substantia nigra pars compacta dopaminergic neuron (Canavier/Komendantov
kinetics). Carries dopaminergic sodium and potassium channels, a low-voltage-
activated Cav1.3 L-type calcium current, an SNc-tuned persistent sodium current,
an SK calcium-activated potassium current, and Ih, with calcium dynamics. Fires
a slow, regular tonic pacemaker rhythm; Ih drives prominent sag and an Ih-
mediated rebound spike. This single-compartment model has no T-type calcium
current and does not reproduce depolarization block.

### Thalamic Relay Neuron

A thalamocortical relay neuron (McCormick & Huguenard 1992). Carries relay-tuned
sodium and potassium, a slow-inactivating T-type calcium current (ICaT), and Ih,
with calcium dynamics. Exhibits the two canonical thalamic modes: tonic firing
from depolarized potentials, and — after sustained hyperpolarization that de-
inactivates ICaT — a post-inhibitory low-threshold calcium spike (LTS) crowned
by a burst of fast spikes on release.

### Hippocampal CA1 Pyramidal Neuron

A hippocampal CA1 pyramidal cell. The most channel-dense preset: Nav1.2 sodium,
Pospischil potassium, A-type potassium (IKa), muscarinic IM, persistent sodium
INaP, L-, N-, and T-type calcium (ICaL, ICaN, ICaT), calcium-activated potassium
IKCa, and Ih, with calcium dynamics. Fires regular trains with strong spike-
frequency adaptation driven by IM accumulation and gradual IKCa activation.

### Subthalamic Nucleus Neuron

A subthalamic nucleus (STN) pacemaker (Bevan & Wilson 1999). Carries STN sodium,
Kv3.1, A-type potassium, persistent sodium, L- and T-type calcium, calcium-
activated potassium IKCa, an ATP-sensitive potassium current (KATP), and Ih,
with calcium dynamics. Fires an autonomous tonic rhythm (about 5-50 Hz) and
produces a post-inhibitory rebound burst via its strong T-type calcium current.

### Thalamic Reticular Nucleus Neuron

A thalamic reticular nucleus (TRN) neuron (Huguenard & Prince 1992). Carries
TRN-tuned sodium and potassium, a T-type calcium current, calcium-activated
potassium IKCa, and Ih, with calcium dynamics. Fires a tonic pacemaker rhythm
(around 10 Hz) and produces a pronounced post-inhibitory LTS rebound burst (5-15
spikes) on release from hyperpolarization.

## Customizing a preset

Because a `Neuron` is immutable, customizing one means building a new instance.
Start from a factory to get a realistic baseline, then construct a modified
`Neuron` (for example with different conductances, ion concentrations, or
temperature). See the `Neuron` entry in the API reference for the full set of
constructor parameters and the channel factory functions available under
`patch_sim.channels`.
