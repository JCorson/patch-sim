"""Protocol parameter state for the patch_sim web UI."""

import logging

import numpy as np
import reflex as rx

import patch_sim
import patch_sim.clamp_simulations
from patch_sim.constants import CURRENT_CLAMP
from patch_sim.presets import NEURON_PROTOCOL_ADJUSTMENTS, PROTOCOL_PRESETS
from patch_sim.protocols.builders import build_current_protocol, build_voltage_protocol
from patch_sim_ui import constants, presets
from patch_sim_ui.state._common import _make_float_setter, _set_float

_PROTOCOL_FLOAT_FIELDS: list[str] = [
    # Shared timing
    "pre_stimulus_duration",
    "stimulus_duration",
    "post_stimulus_duration",
    # Current clamp protocol params
    "start_current",
    "end_current",
    "pulse_amplitude",
    "pulse_width",
    "pulse_interval",
    "dc_offset",
    "amplitude",
    "frequency",
    "start_frequency",
    "end_frequency",
    "mean_current",
    "std_current",
    # Voltage clamp protocol params
    "holding_voltage",
    "vc_start_voltage",
    "vc_end_voltage",
    "vc_pulse_amplitude",
    "vc_pulse_width",
    "vc_pulse_interval",
]

logger = logging.getLogger(__name__)


class ProtocolState(rx.State):
    """State for experiment mode, protocol parameters, and protocol loading."""

    # ------------------------------------------------------------------ #
    # Experiment mode                                                     #
    # ------------------------------------------------------------------ #
    clamp_mode: str = CURRENT_CLAMP  # CURRENT_CLAMP | VOLTAGE_CLAMP

    # ------------------------------------------------------------------ #
    # Protocol parameters                                                #
    # ------------------------------------------------------------------ #
    protocol_type: str = "Step"
    pre_stimulus_duration: float = 10.0
    stimulus_duration: float = 30.0
    post_stimulus_duration: float = 10.0

    # Stimulus amplitude params — shared (units depend on clamp_mode)
    min_stimulus: float = 10.0
    max_stimulus: float = 10.0
    stimulus_step: float = 0.0

    # Current clamp protocol params
    start_current: float = 0.0
    end_current: float = 15.0
    pulse_amplitude: float = 10.0
    pulse_width: float = 2.0
    pulse_interval: float = 10.0
    dc_offset: float = 8.0
    amplitude: float = 4.0
    frequency: float = 50.0
    start_frequency: float = 1.0
    end_frequency: float = 100.0
    mean_current: float = 8.0
    std_current: float = 2.0

    # Voltage clamp protocol params
    holding_voltage: float = -70.0
    vc_start_voltage: float = -70.0
    vc_end_voltage: float = 40.0
    vc_pulse_amplitude: float = 20.0
    vc_pulse_width: float = 2.0
    vc_pulse_interval: float = 10.0

    # ------------------------------------------------------------------ #
    # Preset tracking                                                     #
    # ------------------------------------------------------------------ #
    active_protocol_preset: str = ""

    # ------------------------------------------------------------------ #
    # Computed properties                                                #
    # ------------------------------------------------------------------ #
    @rx.var
    def protocol_options(self) -> list[str]:
        """Protocol type options filtered by clamp mode."""
        if self.clamp_mode == CURRENT_CLAMP:
            return constants.CURRENT_PROTOCOLS
        return constants.VOLTAGE_PROTOCOLS

    @rx.var
    def is_step_single_sweep(self) -> bool:
        """True when the Step protocol is configured as a single sweep.

        A single sweep is produced whenever min_stimulus == max_stimulus,
        regardless of the stimulus_step value.

        Returns:
            True if min_stimulus equals max_stimulus, False otherwise.
        """
        return self.min_stimulus == self.max_stimulus

    @rx.var
    def can_run_continuous(self) -> bool:
        """True when the active protocol is compatible with continuous mode.

        Multi-sweep Step configurations (min_stimulus != max_stimulus) are
        excluded; all other protocols run as a single sweep and are compatible.
        """
        if self.protocol_type != "Step":
            return True
        return self.is_step_single_sweep

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def set_protocol_type(self, value: str) -> None:
        """Set protocol_type from a select event.

        Args:
            value: New protocol type string.
        """
        self.protocol_type = value

    def _apply_clamp_mode(self, mode: str) -> None:
        """Apply clamp mode and reset protocol type synchronously.

        Sets ``clamp_mode`` and resets ``protocol_type`` to the first option
        for the new mode.  Does not touch cross-state fields (use
        :meth:`set_clamp_mode` for the full async handler).

        Args:
            mode: New clamp mode string (``CURRENT_CLAMP`` or ``VOLTAGE_CLAMP``).
        """
        self.clamp_mode = mode
        if mode == CURRENT_CLAMP:
            self.protocol_type = constants.CURRENT_PROTOCOLS[0]
        else:
            self.protocol_type = constants.VOLTAGE_PROTOCOLS[0]

    async def set_clamp_mode(self, mode: str) -> None:
        """Switch between Current Clamp and Voltage Clamp modes.

        Resets ``protocol_type`` to the first option for the new mode and
        clears all simulation results in SimulationState.  Syncs the
        ``_figure_clamp_mode`` shadow copy so that SimulationState's ``figure_data``
        computed var stays consistent.

        Args:
            mode: New clamp mode string (``CURRENT_CLAMP`` or ``VOLTAGE_CLAMP``).
        """
        from patch_sim_ui.state.simulation import SimulationState

        self._apply_clamp_mode(mode)
        sim_st = await self.get_state(SimulationState)
        sim_st._clear_for_new_protocol()
        sim_st._figure_clamp_mode = mode

    def _apply_protocol_preset(self, name: str, neuron_type: str = "") -> None:
        """Apply protocol preset parameters synchronously.

        Sets all protocol parameters for the given preset and overlays any
        neuron-type-specific adjustments.  Does not clear sweep collections
        or touch cross-state fields.

        Args:
            name: Key into PROTOCOL_PRESETS.  Silently ignored if not found.
            neuron_type: Active neuron type name used to look up adjustments in
                NEURON_PROTOCOL_ADJUSTMENTS.  Pass an empty string to skip.
        """
        if name not in PROTOCOL_PRESETS:
            return
        config = dict(PROTOCOL_PRESETS[name])
        adjustments = NEURON_PROTOCOL_ADJUSTMENTS.get(neuron_type, {}).get(name, {})
        config.update(adjustments)
        for key, value in config.items():
            setattr(self, key, value)
        self.active_protocol_preset = name

    async def load_protocol_preset(self, name: str) -> None:
        """Load a protocol preset, applying neuron-type adjustments if active.

        Applies the base protocol preset, overlays neuron-type-specific
        adjustments, records the preset name in ``active_protocol_preset``,
        clears simulation results in SimulationState, and syncs the
        ``_figure_clamp_mode`` shadow copy.

        Args:
            name: Key into PROTOCOL_PRESETS.  Ignored if not found.
        """
        from patch_sim_ui.state.neuron import NeuronState
        from patch_sim_ui.state.simulation import SimulationState

        if name not in PROTOCOL_PRESETS:
            logger.debug("load_protocol_preset: unknown preset %r ignored", name)
            return
        logger.info("Loaded protocol preset: %s", name)
        neuron_st = await self.get_state(NeuronState)
        self._apply_protocol_preset(name, neuron_st.active_neuron_type)
        for key, value in presets.PROTOCOL_NEURON_OVERRIDES.get(name, {}).items():
            setattr(neuron_st, key, value)
        sim_st = await self.get_state(SimulationState)
        sim_st._clear_for_new_protocol()
        sim_st._figure_clamp_mode = self.clamp_mode

    # ------------------------------------------------------------------ #
    # Numeric field setters                                              #
    # ------------------------------------------------------------------ #
    def _set_float(self, field: str, value: "str | list[float] | float") -> None:
        """Coerce value to float and set the named field.

        Args:
            field: Name of the ProtocolState attribute to update.
            value: Raw value from an input or slider event.
        """
        _set_float(self, field, value)

    # One setter per protocol float field — generated at class-definition time
    # so Reflex's metaclass sees them as regular event handlers.
    for _f in _PROTOCOL_FLOAT_FIELDS:
        vars()[f"set_{_f}"] = _make_float_setter(_f, "ProtocolState")

    def set_min_stimulus(self, value: "str | float") -> None:
        """Set min_stimulus, auto-setting stimulus_step when a range is opened.

        If the new min_stimulus differs from max_stimulus and stimulus_step is
        currently 0, stimulus_step is set to 1.0 so the Step protocol remains
        in a valid multi-sweep state without requiring a separate user action.

        Args:
            value: Raw input value from the UI field.
        """
        self._set_float("min_stimulus", value)
        if self.min_stimulus != self.max_stimulus and self.stimulus_step == 0.0:
            self.stimulus_step = 1.0

    def set_max_stimulus(self, value: "str | float") -> None:
        """Set max_stimulus, auto-setting stimulus_step when a range is opened.

        If the new max_stimulus differs from min_stimulus and stimulus_step is
        currently 0, stimulus_step is set to 1.0 so the Step protocol remains
        in a valid multi-sweep state without requiring a separate user action.

        Args:
            value: Raw input value from the UI field.
        """
        self._set_float("max_stimulus", value)
        if self.min_stimulus != self.max_stimulus and self.stimulus_step == 0.0:
            self.stimulus_step = 1.0

    def set_stimulus_step(self, value: "str | float") -> None:
        """Set stimulus_step, rejecting non-positive values when min != max.

        When min_stimulus differs from max_stimulus a step of 0 (or negative)
        would produce an invalid multi-sweep configuration.  Such values are
        ignored, leaving the previous step unchanged.  When
        min_stimulus == max_stimulus (single-sweep mode) any value is accepted.

        Args:
            value: Raw input value from the UI text field.
        """
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            logger.debug("set_stimulus_step: could not parse %r as float", value)
            return
        if self.min_stimulus != self.max_stimulus and parsed <= 0.0:
            logger.debug(
                "set_stimulus_step: rejected value %s"
                " (non-positive step in multi-sweep mode)",
                parsed,
            )
            self.stimulus_step = 1.0
            return
        self.stimulus_step = parsed

    # ------------------------------------------------------------------ #
    # Protocol building                                                  #
    # ------------------------------------------------------------------ #
    def _attach_step_labels(
        self, arrays: "np.ndarray", fmt: str
    ) -> "list[tuple[np.ndarray, str]]":
        """Pair rows of a 2-D stimulus array with formatted step labels.

        Re-derives the stimulus values from ``min_stimulus``, ``max_stimulus``,
        and ``stimulus_step`` using the same formula as the builder, so the
        labels always match the arrays.

        Args:
            arrays: 2-D stimulus array of shape ``(n_sweeps, n_samples)``
                returned by a builder function.
            fmt: Format string applied to each stimulus value (e.g.
                ``"{:+.1f} µA/cm²"``).

        Returns:
            List of (array, label) pairs with one entry per sweep.
        """
        n_steps = (
            round((self.max_stimulus - self.min_stimulus) / self.stimulus_step) + 1
        )
        values = np.linspace(self.min_stimulus, self.max_stimulus, n_steps)
        return [(row, fmt.format(v)) for row, v in zip(arrays, values)]

    def _build_protocols(self) -> "list[tuple[np.ndarray, str]]":
        """Build stimulus arrays from current protocol state with sweep labels.

        Labels are generated here in the UI layer: multi-sweep Step protocols
        produce descriptive labels per sweep; all other protocols use an empty
        string.

        Returns:
            List of (stimulus_array, sweep_label) pairs. Single-sweep protocols
            return a one-element list with an empty label; multi-sweep protocols
            (e.g. I-V Curve) return one entry per sweep with a descriptive label.
        """
        fs = patch_sim.clamp_simulations.SIM_SAMPLING_FREQ
        if self.clamp_mode == "Current Clamp":
            arrays = build_current_protocol(
                protocol_type=self.protocol_type,
                sampling_frequency=fs,
                pre_stimulus_duration=self.pre_stimulus_duration,
                stimulus_duration=self.stimulus_duration,
                post_stimulus_duration=self.post_stimulus_duration,
                min_stimulus=self.min_stimulus,
                max_stimulus=self.max_stimulus,
                stimulus_step=self.stimulus_step,
                start_current=self.start_current,
                end_current=self.end_current,
                pulse_amplitude=self.pulse_amplitude,
                pulse_width=self.pulse_width,
                pulse_interval=self.pulse_interval,
                dc_offset=self.dc_offset,
                amplitude=self.amplitude,
                frequency=self.frequency,
                start_frequency=self.start_frequency,
                end_frequency=self.end_frequency,
                mean_current=self.mean_current,
                std_current=self.std_current,
            )
            if arrays.shape[0] > 1:
                return self._attach_step_labels(arrays, "{:+.1f} µA/cm²")
            return [(arrays[0], "")]
        else:
            arrays = build_voltage_protocol(
                protocol_type=self.protocol_type,
                sampling_frequency=fs,
                pre_stimulus_duration=self.pre_stimulus_duration,
                stimulus_duration=self.stimulus_duration,
                post_stimulus_duration=self.post_stimulus_duration,
                holding_voltage=self.holding_voltage,
                start_voltage=self.vc_start_voltage,
                end_voltage=self.vc_end_voltage,
                pulse_amplitude=self.vc_pulse_amplitude,
                pulse_width=self.vc_pulse_width,
                pulse_interval=self.vc_pulse_interval,
                min_stimulus=self.min_stimulus,
                max_stimulus=self.max_stimulus,
                stimulus_step=self.stimulus_step,
            )
            if arrays.shape[0] > 1:
                return self._attach_step_labels(arrays, "{:+.0f} mV")
            return [(arrays[0], "")]
