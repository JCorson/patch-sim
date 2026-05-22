# Protocols

Stimulus builders for current- and voltage-clamp simulations. The dispatchers
(`build_current_protocol`, `build_voltage_protocol`) select a generator by name;
the generators can also be called directly for single-sweep stimuli.

## Dispatchers

::: patch_sim.build_current_protocol

::: patch_sim.build_voltage_protocol

## Current-clamp generators

::: patch_sim.step_current

::: patch_sim.ramp_current

::: patch_sim.pulse_train

::: patch_sim.sinusoidal_current

::: patch_sim.chirp_current

::: patch_sim.noise_current

## Voltage-clamp generators

::: patch_sim.step_voltage

::: patch_sim.ramp_voltage

::: patch_sim.pulse_train_voltage
