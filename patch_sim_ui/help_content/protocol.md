# Setting up a protocol

The **Experiment** panel (lower left sidebar) defines the stimulus applied to
the neuron.

![The Experiment panel](/screenshots/protocol-panel.png)

## Protocol presets

The *Load preset…* dropdown loads a ready-made experiment, tuned to the selected
neuron. The presets are:

- **Action Potential** — a brief step that evokes a single spike.
- **Subthreshold Response** — a step that stays just below threshold.
- **Repetitive Firing** — a sustained step that drives a spike train.
- **F-I Curve** — a multi-sweep series of increasing current steps.
- **Hyperpolarization Steps** — negative steps revealing sag and rebound.
- **I-V Curve** — a voltage-clamp series for a current-voltage relationship.
- **Na+ Channel Activation** — a voltage-clamp activation protocol.
- **Steady-State Inactivation** — a two-pulse voltage-clamp inactivation protocol.
- **Frequency Response** — a swept-frequency (chirp) current for impedance.

## Mode

The **Mode** radio group selects the clamp type:

- **Current Clamp** — inject a current and record the membrane voltage.
- **Voltage Clamp** — hold a commanded voltage and record the ionic current.

## Protocol type

The **Protocol** dropdown lists the waveforms available for the current mode:

- **Current Clamp** — Step, Ramp, Pulse Train, Sinusoidal, Chirp, Noise.
- **Voltage Clamp** — Step, Ramp, Pulse Train, Inactivation.

## Timing and parameters

Three timing fields apply to every protocol:

- **Pre-stimulus (ms)** — baseline before the stimulus.
- **Stimulus (ms)** — duration of the stimulus.
- **Post-stimulus (ms)** — recovery after the stimulus.

Below them, the panel shows fields specific to the chosen protocol (for example
*Current (µA/cm²)* for a step, or *Start freq (Hz)* / *End freq (Hz)* for a
chirp).

## Multi-sweep protocols

Step protocols include a **Multi-sweep** toggle. With it on, you set a range
(*min*, *max*, *step*) and the simulation runs one sweep per level — this is how
F-I and I-V series are produced. Each sweep is drawn as a separate colored trace
in the plot.
