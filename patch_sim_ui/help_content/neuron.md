# Configuring the neuron

The **Neuron Parameters** panel (top of the left sidebar) defines the cell
model: which ion channels it has, its membrane properties, and its ion
concentrations.

![The Neuron Parameters panel](/screenshots/neuron-panel.png)

## Presets

The *Load neuron type…* dropdown loads a fully configured cell model. Nine
presets are available:

- **Squid Giant Axon (Classic HH)** — the original Hodgkin-Huxley model.
- **Fast-Spiking Interneuron** — non-adapting high-frequency firing.
- **Cortical Pyramidal Neuron** — regular spiking with adaptation.
- **Purkinje Neuron** — spontaneous tonic pacemaker.
- **SNc Dopaminergic Neuron** — slow tonic pacemaker.
- **Thalamic Relay Neuron** — tonic firing and rebound bursts.
- **Hippocampal CA1 Pyramidal Neuron** — strong spike-frequency adaptation.
- **Subthalamic Nucleus Neuron** — autonomous pacemaker.
- **Thalamic Reticular Nucleus Neuron** — tonic firing with rebound bursts.

Loading a preset replaces all parameters below; you can then adjust any of them.

## Membrane Properties

Expand the **Membrane Properties** accordion to set:

- **Capacitance (µF/cm²)** — membrane capacitance.
- **Resting potential (mV)** — the leak reversal that sets the resting voltage.
- **Temperature (K)** — sets channel kinetics via Q10 scaling.

Each field has both a numeric input and a slider.

## Ion Concentrations (mM)

Expand **Ion Concentrations** to set the intra- and extracellular concentrations
of **Na⁺**, **K⁺**, and **Ca²⁺**. These determine the reversal potentials via
the Nernst equation, so changing them shifts the driving forces on each current.

## Channels

When the selected preset includes additional (auxiliary) channels, a
**Channels** accordion appears with a **Max conductance (mS/cm²)** slider for
each one (for example *Ih (HCN)*, *IKa (A-type K⁺)*, *IKv31 (Kv3.1-type K⁺)*).
Set a conductance to zero to switch a channel off.

## Reversal Potentials

The read-only **Reversal Potentials** grid shows the computed equilibrium
potentials for **Na⁺**, **K⁺**, **Leak**, and **Ca²⁺** (in mV), updating live as
you change concentrations or temperature.
