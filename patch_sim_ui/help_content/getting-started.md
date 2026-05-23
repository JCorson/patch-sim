# Getting started

The Patch Clamp Simulator lets you run virtual patch-clamp experiments in the
browser: pick a neuron, apply a stimulus, and watch the membrane respond — no
coding required.

![The simulator after running an action-potential protocol](/screenshots/overview.png)

## The layout

The window has three regions:

- **Left sidebar** — two panels: **Neuron Parameters** (the cell model) and
  **Experiment** (the stimulus protocol).
- **Center** — the **trace plot**, the sweep controls beneath it, and a
  collapsible **Logs** panel.
- **Right sidebar** — the **Analysis** panel, showing metrics for the current
  result.

## The basic loop

1. **Choose a neuron.** In **Neuron Parameters**, pick a cell type from the
   *Load neuron type…* dropdown (or tune parameters by hand).
2. **Choose a protocol.** In **Experiment**, pick a preset from the *Load
   preset…* dropdown (for example *Action Potential* or *F-I Curve*), or set the
   clamp mode, protocol type, and stimulus parameters yourself.
3. **Run it.** Click **Run** in the top bar. A *Running…* indicator appears
   while the simulation integrates.
4. **Read the result.** The recorded trace appears in the center plot and the
   analysis metrics populate on the right.

## Top-bar controls

- **Reset** — restore all parameters to the defaults for the current neuron and
  protocol.
- **Run** — run a single simulation with the current settings.
- **Continuous** — loop the simulation repeatedly for an oscilloscope-like live
  view; the button becomes **Stop** while running.
- **Help** (the question-mark icon) — open this guide.
- **Theme** — switch between light, dark, and system appearance.

## Next steps

- [Configuring the neuron](neuron.md) — presets, channels, and ion concentrations.
- [Setting up a protocol](protocol.md) — clamp modes, protocol types, and sweeps.
- [Running & reading results](running.md) — the trace plot, sweeps, and stored
  overlays.
- [Analysis panels](analysis.md) — the metrics computed from each run.

For the Python library and API reference, use the **API reference** link at the
top of this page.
