# Patch Clamp Simulator

`patch_sim` simulates patch-clamp electrophysiology experiments using
conductance-based (Hodgkin-Huxley) neuron models. It reproduces how action
potentials arise from the dynamics of voltage- and calcium-gated ion channels,
and ships with an interactive web UI inspired by pClamp.

There are two ways to use it:

- **The web UI** — configure a neuron, pick a stimulus protocol, run it, and
  read live trace plots and analysis metrics. No coding required.
- **The Python library** — build neurons and protocols, run simulations, and
  analyze the results programmatically.

## The workflow

Every experiment, in the UI or in code, follows the same four steps:

1. **Neuron** — choose a cell model (a built-in preset or a custom `Neuron`)
   that defines the membrane, its ion channels, and ion concentrations.
2. **Protocol** — build a stimulus: a current waveform (current clamp) or a
   commanded voltage waveform (voltage clamp).
3. **Simulate** — integrate the Hodgkin-Huxley equations to record the
   response (membrane voltage, per-channel currents, gating variables).
4. **Analyze** — extract metrics: spike trains, F-I curves, I-V curves,
   activation/inactivation curves, passive properties, bursts, and more.

## Architecture at a glance

The project is split into two cleanly separated layers:

- **`patch_sim/`** — the pure-Python core library. All simulation, channel,
  protocol, and analysis logic lives here. It has no UI dependency.
- **`patch_sim_ui/`** — the [Reflex](https://reflex.dev) web application that
  wraps the core library in interactive controls and plots.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```
# Core library only
uv sync --frozen

# Core library plus the web UI
uv sync --frozen --group=ui
```

## Running the UI

```
uv run reflex run
```

Then open `http://localhost:3000`. Configure the neuron and protocol in the left
sidebar, press **Run**, and inspect the trace and analysis panels.

## A first simulation in code

The example below builds a classic squid-axon neuron, injects a 50 ms current
step, simulates the response, and counts the action potentials:

```python
import patch_sim
from patch_sim import simulate_current_clamp, step_current

# 1. Neuron: a built-in preset factory returns a configured Neuron.
neuron = patch_sim.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()

# 2. Protocol: a 50 ms, 15 uA/cm^2 current step (a 1-D stimulus array).
stimulus = step_current(duration=50.0, current_amplitude=15.0)

# 3. Simulate: returns a structured array with named fields.
result = simulate_current_clamp(neuron, current_external=stimulus)

# 4. Analyze: detect spikes in the recorded trace.
aps = patch_sim.analyze_aps(result["time"], result["voltage"])
print(aps.spike_count, "spikes; peak", result["voltage"].max(), "mV")
```

`result` is a NumPy structured array. Its fields always include `time`,
`voltage`, the total membrane current `Itotal`, the per-channel currents (for
the squid preset, `INa`, `IK`, `INaL`, `IKL`), and the gating variables
(`m`, `h`, `n`). `len(result)` equals the number of samples in the stimulus.

## Core concepts

- **`Neuron`** — an immutable model holding a tuple of ion channels, membrane
  capacitance, ion concentrations, temperature, and an optional `CalciumDynamics`
  model. Reversal potentials are derived from the Nernst (or Goldman) equation.
- **Channels and gating** — each ion channel carries one or more gating
  variables whose voltage- (or calcium-) dependent kinetics open and close the
  channel over time.
- **Current vs. voltage clamp** — current clamp injects current and records
  voltage; voltage clamp holds a commanded voltage and records the resulting
  ionic currents.
- **Protocols** — stimulus builders that produce the input waveform for a
  simulation (single-sweep arrays from the low-level builders; multi-sweep
  arrays from the preset dispatcher).
- **Analysis** — post-hoc functions that turn a `SimulationResult` into
  physiological metrics.

## Where to go next

- [Neuron presets](presets.md) — the nine built-in cell types and their biology.
- [Protocols & analysis](protocols-and-analysis.md) — the stimulus library and
  the analysis metrics.
- **API reference** — the full auto-generated documentation for every public
  symbol (see the API reference section of this site).
