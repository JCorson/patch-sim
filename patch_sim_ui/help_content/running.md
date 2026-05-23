# Running & reading results

## Running a simulation

Click **Run** in the top bar to simulate once with the current settings. While
it computes, the bar shows a *Running…* indicator; when it finishes, the trace
and analysis update.

**Continuous** loops the simulation for a live, oscilloscope-like view (useful
while dragging a slider). Click **Stop** to end it. **Reset** returns every
parameter to the defaults for the current neuron and protocol.

## The trace plot

The center plot shows the recorded response against time (ms):

- **Current clamp** plots membrane **voltage (mV)**.
- **Voltage clamp** plots **current (µA/cm²)**.

![A simulated action-potential trace](/screenshots/trace-plot.png)

Multi-sweep protocols (F-I, I-V, hyperpolarization steps) overlay one colored
trace per sweep:

![A multi-sweep F-I protocol](/screenshots/fi-curve.png)

## Sweep and trace controls

The controls beneath the plot manage what is shown:

![The sweep and trace controls](/screenshots/sweep-manager.png)

- **Traces** (the eye icon) opens a popover of checkboxes to show or hide
  individual traces — the voltage/current, per-channel currents, and gating
  variables available for the current model and mode.
- The **crosshair** icon toggles hover tooltips on the plot.
- **Store** saves the current trace as a translucent reference overlay (labeled
  *Stored 1*, *Stored 2*, …) so you can compare it against later runs.
  **Clear Stored** removes the overlays. Storing is unavailable for multi-sweep
  protocols.
- **Logs** opens a panel of simulation messages (warnings, errors, debug). You
  can filter by level and clear it.
