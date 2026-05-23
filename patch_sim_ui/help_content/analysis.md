# Analysis panels

The **Analysis** panel on the right computes metrics from the latest run. Its
contents depend on the clamp mode and protocol. Use the close icon to collapse
it to a thin strip, and the icon on that strip to reopen it.

![The analysis panel after an action-potential run](/screenshots/analysis.png)

## Passive membrane test

Shown for every run: **Input resistance**, **Time constant**, and
**Capacitance**, measured from a small built-in test pulse.

## Current clamp

For most current-clamp protocols the panel shows **AP Analysis**:

- Summary metrics — **Spikes**, **Threshold**, **Peak**, **Rise time**,
  **Half-width**, **AHP depth**, and **Rheobase** (plus **ISI**, **Firing
  rate**, and **Adapt. index** for single sweeps).
- A per-spike table with threshold, peak, rise time, half-width, and AHP depth.
- A **phase-plane** plot (dV/dt vs. V).
- For multi-sweep runs, an **F-I curve** (firing rate vs. injected current).

Two further tabs are available:

- **Bursts** — burst count, spikes per burst, intra-burst frequency, inter-burst
  interval, and a per-burst table.
- **Calcium** — transient peak, decay, and return time (for models with calcium
  dynamics).

## Chirp protocol

When the protocol is the chirp (Frequency Response), the panel shows
**Impedance Analysis**: the impedance magnitude versus frequency, with the
resonance frequency and Q factor.

## Voltage clamp

In voltage clamp the panel shows **I-V Analysis**:

- An **I-V curve** (peak inward, peak outward, and steady-state current vs.
  voltage).
- A **g-V** activation curve with a Boltzmann fit.
- An **h∞** steady-state inactivation curve (for the Inactivation protocol).
- a **τ-V** plot of activation/inactivation time constants where available.
