# Protocols & analysis

This page covers the second and fourth steps of the workflow: building stimulus
**protocols** and running **analysis** on the recorded results.

## Clamp modes

A simulation runs in one of two modes, identified by the constants
`patch_sim.CURRENT_CLAMP` (`"Current Clamp"`) and `patch_sim.VOLTAGE_CLAMP`
(`"Voltage Clamp"`):

- **Current clamp** injects a current waveform and records the resulting
  membrane voltage (and the underlying channel currents and gating variables).
- **Voltage clamp** holds the membrane at a commanded voltage waveform and
  records the ionic currents required to clamp it.

## Current-clamp protocols

The available current-clamp protocol types are listed in
`patch_sim.CURRENT_PROTOCOLS`:

| Type | Builder | Description |
| --- | --- | --- |
| Step | `step_current` | A constant current step between two times. |
| Ramp | `ramp_current` | A linearly increasing (or decreasing) current. |
| Pulse Train | `pulse_train` | A repeating train of brief current pulses. |
| Sinusoidal | `sinusoidal_current` | A sine wave on a DC offset. |
| Chirp | `chirp_current` | A swept-frequency sine (ZAP) for impedance probing. |
| Noise | `noise_current` | Gaussian white-noise current. |

The low-level builders (`step_current`, `ramp_current`, ...) return a single
**1-D** stimulus array ready to pass straight to `simulate_current_clamp`. The
higher-level `build_current_protocol(protocol_type, sampling_frequency, ...)`
dispatches to them by name with a uniform keyword interface.

## Voltage-clamp protocols

The available voltage-clamp protocol types are listed in
`patch_sim.VOLTAGE_PROTOCOLS`:

| Type | Builder | Description |
| --- | --- | --- |
| Step | `step_voltage` | A holding potential stepped to a test potential. |
| Ramp | `ramp_voltage` | A linear voltage ramp (for I-V curves). |
| Pulse Train | `pulse_train_voltage` | A repeating train of voltage pulses. |
| Inactivation | (two-pulse protocol) | A conditioning prepulse then a fixed test pulse, for steady-state inactivation (h-infinity). |

The dispatcher is `build_voltage_protocol(protocol_type, ...)`. The
steady-state-inactivation protocol type is named by
`patch_sim.INACTIVATION_PROTOCOL` (`"Inactivation"`).

## Protocol presets

For ready-made experiments, `build_protocol_from_preset` composes a named
protocol preset with a neuron's per-cell adjustments:

```python
import patch_sim

# (protocol preset name, neuron preset name)
protocol = patch_sim.build_protocol_from_preset(
    "F-I Curve", "Cortical Pyramidal Neuron"
)
```

The protocol-preset names are in `patch_sim.PROTOCOL_PRESET_NAMES`: **Action
Potential**, **Subthreshold Response**, **Repetitive Firing**, **F-I Curve**,
**Hyperpolarization Steps**, **I-V Curve**, **Na+ Channel Activation**,
**Steady-State Inactivation**, and **Frequency Response**. These are the presets
exposed in the UI's protocol picker.

Unlike the single-sweep builders, `build_protocol_from_preset` returns a **2-D**
array shaped (sweeps, samples) — multi-sweep protocols such as an F-I or I-V
series produce one row per sweep, driven by the multi-sweep runner.

## Running simulations

The two core runners take a `Neuron` and a stimulus array:

```python
result = patch_sim.simulate_current_clamp(neuron, current_external=stimulus)
result = patch_sim.simulate_voltage_clamp(neuron, voltage_protocol=command)
```

Both return a `SimulationResult` — a NumPy structured array whose fields include
`time`, `voltage`, the total membrane current `Itotal`, one current field per
channel, and the gating variables. Related entry points:

- `simulate_current_clamp_from_state` / `simulate_voltage_clamp_from_state` —
  resume a simulation from a saved gating state (used for continuous mode).
- `simulate_batch` — run many protocols efficiently.

## Analysis

The `patch_sim.analysis` subpackage turns a recorded trace into physiological
metrics. Each family has a high-level entry point and a typed result object.

- **Action potentials** — `analyze_aps(time, voltage)` returns an
  `APAnalysisResult` (`spike_count`, `firing_rate`, per-spike `SpikeMetrics`,
  inter-spike intervals, mean peak/threshold/half-width/AHP).
- **F-I curves** — `analyze_fi` builds a firing-rate-vs-current curve;
  `estimate_rheobase` finds the firing threshold; `compute_fi_point` scores one
  sweep.
- **I-V curves** — `analyze_iv` / `compute_iv_point` build a current-vs-voltage
  curve from a voltage-clamp series.
- **Activation (G-V) curves** — `compute_gv` builds a conductance-vs-voltage
  curve and `boltzmann` fits it, returning a `BoltzmannFit`.
- **Steady-state inactivation** — `compute_inactivation` builds the h-infinity
  curve from the two-pulse inactivation protocol.
- **Bursting** — `analyze_bursts` detects bursts and reports `BurstMetrics`
  (intra-burst rate, spikes per burst, burst duration).
- **Calcium transients** — `analyze_calcium_transients` detects calcium
  transients and fits their decay (for presets with calcium dynamics).
- **Sag and rebound** — `analyze_hyperpolarization` / `compute_sag_point`
  quantify Ih-driven sag and post-inhibitory rebound.
- **Impedance** — `analyze_impedance` computes an impedance profile from a chirp
  (ZAP) protocol; `impedance_unavailable_reason` explains when it cannot.
- **Spike-frequency adaptation** — `analyze_sfa` / `compute_sfa` quantify ISI
  growth across a sustained train.
- **Membrane time constant (tau-V)** — `analyze_tau_v` / `compute_tau_v_point`
  fit exponential relaxations from voltage-clamp steps.
- **Passive properties** — `analyze_passive_properties` and `run_membrane_test`
  measure input resistance, capacitance, and membrane time constant, returning
  `PassiveProperties`.

A minimal current-clamp-then-analyze example is in the
[overview](index.md#a-first-simulation-in-code). For full signatures and the
result-object fields, see the **API reference** section of this site.
