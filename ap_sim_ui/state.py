"""Application state for the ap_sim web UI.

All reactive variables and event handlers live here. The state drives
the Reflex component tree via computed properties.
"""

import numpy as np
import plotly.graph_objects as go
import reflex as rx

import ap_sim
import ap_sim.clamp_simulations
from ap_sim.constants import (
    DEFAULT_C_M,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
    DEFAULT_G_IH,
    DEFAULT_G_IKA,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
)
from ap_sim.additional_channels import make_ih_channel, make_ika_channel
from ap_sim_ui import constants, presets
from ap_sim_ui.plotting import Sweep, build_figure
from ap_sim_ui.protocol_builders import build_current_protocol, build_voltage_protocol

# ------------------------------------------------------------------ #
# Float setter generation                                            #
# ------------------------------------------------------------------ #
# Reflex's auto-generated set_X handlers are typed `value: float`,   #
# but rx.input.on_change passes str and rx.slider.on_change passes   #
# list[float].  The generated setters below coerce both via          #
# AppState._set_float.                                               #

_FLOAT_FIELDS: list[str] = [
    # Neuron parameters
    "g_Na",
    "g_K",
    "g_L",
    "C_m",
    "v_rest",
    "Na_out",
    "Na_in",
    "K_out",
    "K_in",
    "Cl_out",
    "Cl_in",
    "T",
    # Shared protocol params
    "duration",
    # Current clamp protocol params
    "current_amplitude",
    "step_start",
    "step_duration",
    "start_current",
    "end_current",
    "ramp_start",
    "ramp_duration",
    "pulse_amplitude",
    "pulse_width",
    "pulse_interval",
    "train_start",
    "dc_offset",
    "amplitude",
    "frequency",
    "start_frequency",
    "end_frequency",
    "mean_current",
    "std_current",
    # Voltage clamp protocol params
    "vc_voltage_amplitude",
    "vc_step_start",
    "vc_step_duration",
    "vc_holding_voltage",
    "vc_start_voltage",
    "vc_end_voltage",
    "vc_ramp_start",
    "vc_ramp_duration",
    "vc_pulse_amplitude",
    "vc_pulse_width",
    "vc_pulse_interval",
    "vc_train_start",
    "vc_voltage_min",
    "vc_voltage_max",
    "vc_voltage_step",
    "vc_pre_pulse_duration",
    "vc_post_pulse_duration",
    "vc_prepulse_voltage",
    "vc_prepulse_duration",
    "vc_test_voltage_min",
    "vc_test_voltage_max",
    "vc_interpulse_duration",
    # Optional channel params
    "ih_g_max",
    "ika_g_max",
]


_BOOL_FIELDS: list[str] = [
    "show_voltage",
    "show_total_current",
    "show_sodium_current",
    "show_potassium_current",
    "show_leak_current",
    "show_potassium_activation",
    "show_sodium_activation",
    "show_sodium_inactivation",
    # Optional channel visibility
    "ih_enabled",
    "show_ih_current",
    "show_ih_gating",
    "ika_enabled",
    "show_ika_current",
    "show_ika_gating",
]


def _make_bool_setter(field_name: str):
    """Factory returning a bool event handler for ``field_name``."""

    def setter(self, value: bool) -> None:
        setattr(self, field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"AppState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from a checkbox event."
    return setter


def _make_float_setter(field_name: str):
    """Factory returning a float-coercing event handler for ``field_name``.

    Args:
        field_name: Name of the AppState attribute to update.

    Returns:
        An event handler method that accepts ``str | list[float] | float``
        and delegates to ``AppState._set_float``.
    """

    def setter(self, value: "str | list[float] | float") -> None:
        """Set the field from an input or slider event."""
        self._set_float(field_name, value)

    setter.__name__ = f"set_{field_name}"
    setter.__qualname__ = f"AppState.set_{field_name}"
    setter.__doc__ = f"Set {field_name} from an input or slider event."
    return setter


class AppState(rx.State):
    """Top-level application state."""

    # ------------------------------------------------------------------ #
    # Neuron parameters                                                   #
    # ------------------------------------------------------------------ #
    g_Na: float = DEFAULT_G_NA
    g_K: float = DEFAULT_G_K
    g_L: float = DEFAULT_G_L
    C_m: float = DEFAULT_C_M
    v_rest: float = DEFAULT_V_REST
    Na_out: float = DEFAULT_NA_OUT
    Na_in: float = DEFAULT_NA_IN
    K_out: float = DEFAULT_K_OUT
    K_in: float = DEFAULT_K_IN
    Cl_out: float = DEFAULT_CL_OUT
    Cl_in: float = DEFAULT_CL_IN
    T: float = DEFAULT_T

    # ------------------------------------------------------------------ #
    # Optional channels                                                   #
    # ------------------------------------------------------------------ #
    ih_enabled: bool = False
    ih_g_max: float = DEFAULT_G_IH
    ika_enabled: bool = False
    ika_g_max: float = DEFAULT_G_IKA

    # ------------------------------------------------------------------ #
    # Experiment mode                                                     #
    # ------------------------------------------------------------------ #
    clamp_mode: str = "Current Clamp"  # "Current Clamp" | "Voltage Clamp"

    # ------------------------------------------------------------------ #
    # Protocol parameters — shared                                       #
    # ------------------------------------------------------------------ #
    protocol_type: str = "Step"
    duration: float = 50.0

    # Current clamp protocol params
    current_amplitude: float = 10.0
    step_start: float = 10.0
    step_duration: float = 30.0
    start_current: float = 0.0
    end_current: float = 15.0
    ramp_start: float = 0.0
    ramp_duration: float = 40.0
    pulse_amplitude: float = 10.0
    pulse_width: float = 2.0
    pulse_interval: float = 10.0
    train_start: float = 5.0
    dc_offset: float = 8.0
    amplitude: float = 4.0
    frequency: float = 50.0
    start_frequency: float = 1.0
    end_frequency: float = 100.0
    mean_current: float = 8.0
    std_current: float = 2.0

    # Voltage clamp protocol params
    vc_voltage_amplitude: float = 0.0
    vc_step_start: float = 10.0
    vc_step_duration: float = 30.0
    vc_holding_voltage: float = -70.0
    vc_start_voltage: float = -70.0
    vc_end_voltage: float = 40.0
    vc_ramp_start: float = 0.0
    vc_ramp_duration: float = 40.0
    vc_pulse_amplitude: float = 20.0
    vc_pulse_width: float = 2.0
    vc_pulse_interval: float = 10.0
    vc_train_start: float = 5.0
    vc_voltage_min: float = -100.0
    vc_voltage_max: float = 60.0
    vc_voltage_step: float = 10.0
    vc_pre_pulse_duration: float = 5.0
    vc_post_pulse_duration: float = 5.0
    vc_prepulse_voltage: float = -100.0
    vc_prepulse_duration: float = 100.0
    vc_test_voltage_min: float = -60.0
    vc_test_voltage_max: float = 60.0
    vc_interpulse_duration: float = 5.0

    # ------------------------------------------------------------------ #
    # Simulation results                                                  #
    # ------------------------------------------------------------------ #
    current_sweeps: list[Sweep] = []  # Latest simulation result
    saved_sweeps: list[Sweep] = []  # User-saved sweeps for comparison overlay

    # ------------------------------------------------------------------ #
    # Trace visibility checkboxes                                        #
    # ------------------------------------------------------------------ #
    show_voltage: bool = True
    show_total_current: bool = True
    show_sodium_current: bool = True
    show_potassium_current: bool = True
    show_leak_current: bool = False
    show_potassium_activation: bool = False
    show_sodium_activation: bool = False
    show_sodium_inactivation: bool = False
    show_ih_current: bool = True
    show_ih_gating: bool = True
    show_ika_current: bool = True
    show_ika_gating: bool = True

    # ------------------------------------------------------------------ #
    # UI state                                                           #
    # ------------------------------------------------------------------ #
    is_running: bool = False
    error_message: str = ""

    # ------------------------------------------------------------------ #
    # Derived reversal potentials (shown as read-only in neuron panel)  #
    # ------------------------------------------------------------------ #
    @rx.var
    def E_Na(self) -> float:
        """Sodium reversal potential in mV."""
        return float(ap_sim.nernst_potential(1, self.T, self.Na_out, self.Na_in))

    @rx.var
    def E_K(self) -> float:
        """Potassium reversal potential in mV."""
        return float(ap_sim.nernst_potential(1, self.T, self.K_out, self.K_in))

    @rx.var
    def E_L(self) -> float:
        """Leak reversal potential in mV."""
        return float(ap_sim.nernst_potential(-1, self.T, self.Cl_out, self.Cl_in))

    @rx.var
    def protocol_options(self) -> list[str]:
        """Protocol type options filtered by clamp mode."""
        if self.clamp_mode == "Current Clamp":
            return constants.CURRENT_PROTOCOLS
        return constants.VOLTAGE_PROTOCOLS

    @rx.var
    def has_result(self) -> bool:
        """Whether a simulation result is available."""
        return len(self.current_sweeps) > 0

    @rx.var
    def figure_data(self) -> go.Figure:
        """Plotly figure rebuilt whenever relevant state changes."""
        return build_figure(
            current_sweeps=self.current_sweeps,
            saved_sweeps=self.saved_sweeps,
            show_voltage=self.show_voltage,
            show_total_current=self.show_total_current,
            show_sodium_current=self.show_sodium_current,
            show_potassium_current=self.show_potassium_current,
            show_leak_current=self.show_leak_current,
            show_potassium_activation=self.show_potassium_activation,
            show_sodium_activation=self.show_sodium_activation,
            show_sodium_inactivation=self.show_sodium_inactivation,
            clamp_mode=self.clamp_mode,
            show_additional_currents={
                "Ih": self.show_ih_current,
                "IKa": self.show_ika_current,
            },
            show_additional_gating={
                "r": self.show_ih_gating,
                "a": self.show_ika_gating,
                "b": self.show_ika_gating,
            },
        )

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def set_protocol_type(self, value: str) -> None:
        """Set protocol_type from a select event."""
        self.protocol_type = value

    def set_clamp_mode(self, mode: str) -> None:
        """Switch between Current Clamp and Voltage Clamp modes."""
        self.clamp_mode = mode
        # Reset protocol type to first option for the new mode
        if mode == "Current Clamp":
            self.protocol_type = constants.CURRENT_PROTOCOLS[0]
        else:
            self.protocol_type = constants.VOLTAGE_PROTOCOLS[0]

    def load_preset(self, name: str) -> None:
        """Load a named preset configuration."""
        if name not in presets.PRESETS:
            return
        config = presets.PRESETS[name]
        for key, value in config.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------ #
    # Numeric field setters                                              #
    # ------------------------------------------------------------------ #
    def _set_float(self, field: str, value: "str | list[float] | float") -> None:
        """Coerce value to float and set the named field.

        Args:
            field: Name of the AppState attribute to update.
            value: Raw value from an input or slider event.
        """
        v = value[0] if isinstance(value, list) else value
        try:
            setattr(self, field, float(v))
        except (ValueError, TypeError):
            pass

    # One setter per float field — generated at class-definition time so
    # Reflex's metaclass sees them as regular event handlers.
    for _f in _FLOAT_FIELDS:
        vars()[f"set_{_f}"] = _make_float_setter(_f)

    # One setter per bool field — same rationale as float setters above.
    for _f in _BOOL_FIELDS:
        vars()[f"set_{_f}"] = _make_bool_setter(_f)

    # ------------------------------------------------------------------ #
    # Sweep management                                                   #
    # ------------------------------------------------------------------ #
    def add_sweep(self) -> None:
        """Promote current simulation result to the saved sweep overlay."""
        if not self.has_result:
            return
        for sweep in self.current_sweeps:
            idx = len(self.saved_sweeps)
            color = constants.SWEEP_COLORS[idx % len(constants.SWEEP_COLORS)]
            self.saved_sweeps.append(
                sweep.model_copy(update={"color": color, "label": f"Sweep {idx + 1}"})
            )

    def clear_sweeps(self) -> None:
        """Remove all saved sweeps."""
        self.saved_sweeps = []

    @rx.event(background=True)
    async def run_simulation(self) -> None:
        """Build protocol and run the simulation asynchronously.

        Uses background=True so the UI stays responsive during long runs.
        """
        async with self:
            self.is_running = True
            self.error_message = ""

        try:
            opt_channels = []
            if self.ih_enabled:
                opt_channels.append(make_ih_channel(g_max=self.ih_g_max))
            if self.ika_enabled:
                opt_channels.append(make_ika_channel(g_max=self.ika_g_max))
            neuron = ap_sim.HodgkinHuxley(
                g_Na=self.g_Na,
                g_K=self.g_K,
                g_L=self.g_L,
                C_m=self.C_m,
                v_rest=self.v_rest,
                Na_out=self.Na_out,
                Na_in=self.Na_in,
                K_out=self.K_out,
                K_in=self.K_in,
                Cl_out=self.Cl_out,
                Cl_in=self.Cl_in,
                T=self.T,
                additional_channels=tuple(opt_channels),
            )

            fs = ap_sim.clamp_simulations.SIM_SAMPLING_FREQ
            mode = self.clamp_mode
            ptype = self.protocol_type

            if mode == "Current Clamp":
                stimulus = build_current_protocol(
                    protocol_type=ptype,
                    duration=self.duration,
                    sampling_frequency=fs,
                    current_amplitude=self.current_amplitude,
                    step_start=self.step_start,
                    step_duration=self.step_duration,
                    start_current=self.start_current,
                    end_current=self.end_current,
                    ramp_start=self.ramp_start,
                    ramp_duration=self.ramp_duration,
                    pulse_amplitude=self.pulse_amplitude,
                    pulse_width=self.pulse_width,
                    pulse_interval=self.pulse_interval,
                    train_start=self.train_start,
                    dc_offset=self.dc_offset,
                    amplitude=self.amplitude,
                    frequency=self.frequency,
                    start_frequency=self.start_frequency,
                    end_frequency=self.end_frequency,
                    mean_current=self.mean_current,
                    std_current=self.std_current,
                )
                df = ap_sim.simulate_current_clamp(neuron, stimulus)
                async with self:
                    self.current_sweeps = [
                        Sweep.from_dataframe(df, stimulus, "", "", mode)
                    ]

            elif ptype in ("I-V Curve",):
                # Run each voltage step as an independent sweep so that
                # gating variables are reset between steps — matching real
                # patch-clamp I-V curve experiments.
                sweep_duration = (
                    self.vc_pre_pulse_duration
                    + self.duration
                    + self.vc_post_pulse_duration
                )
                voltages = np.arange(
                    self.vc_voltage_min,
                    self.vc_voltage_max + self.vc_voltage_step,
                    self.vc_voltage_step,
                )
                new_sweeps: list[Sweep] = []
                for voltage in voltages:
                    protocol = ap_sim.step_voltage(
                        duration=sweep_duration,
                        voltage_amplitude=float(voltage),
                        step_start=self.vc_pre_pulse_duration,
                        step_duration=self.duration,
                        holding_voltage=self.vc_holding_voltage,
                        sampling_frequency=fs,
                    )
                    sweep_df = ap_sim.simulate_voltage_clamp(neuron, protocol)
                    label = f"{voltage:+.0f} mV"
                    color_index = len(new_sweeps) % len(constants.SWEEP_COLORS)
                    new_sweeps.append(
                        Sweep.from_dataframe(
                            sweep_df,
                            protocol,
                            label,
                            constants.SWEEP_COLORS[color_index],
                            mode,
                        )
                    )
                    async with self:
                        self.current_sweeps = list(new_sweeps)

            else:
                stimulus = build_voltage_protocol(
                    protocol_type=ptype,
                    duration=self.duration,
                    sampling_frequency=fs,
                    vc_holding_voltage=self.vc_holding_voltage,
                    vc_voltage_amplitude=self.vc_voltage_amplitude,
                    vc_step_start=self.vc_step_start,
                    vc_step_duration=self.vc_step_duration,
                    vc_start_voltage=self.vc_start_voltage,
                    vc_end_voltage=self.vc_end_voltage,
                    vc_ramp_start=self.vc_ramp_start,
                    vc_ramp_duration=self.vc_ramp_duration,
                    vc_pulse_amplitude=self.vc_pulse_amplitude,
                    vc_pulse_width=self.vc_pulse_width,
                    vc_pulse_interval=self.vc_pulse_interval,
                    vc_train_start=self.vc_train_start,
                    vc_voltage_min=self.vc_voltage_min,
                    vc_voltage_max=self.vc_voltage_max,
                    vc_voltage_step=self.vc_voltage_step,
                    vc_pre_pulse_duration=self.vc_pre_pulse_duration,
                    vc_post_pulse_duration=self.vc_post_pulse_duration,
                    vc_prepulse_voltage=self.vc_prepulse_voltage,
                    vc_prepulse_duration=self.vc_prepulse_duration,
                    vc_test_voltage_min=self.vc_test_voltage_min,
                    vc_test_voltage_max=self.vc_test_voltage_max,
                    vc_interpulse_duration=self.vc_interpulse_duration,
                )
                df = ap_sim.simulate_voltage_clamp(neuron, stimulus)
                async with self:
                    self.current_sweeps = [
                        Sweep.from_dataframe(df, stimulus, "", "", mode)
                    ]

        except ValueError as exc:
            async with self:
                self.error_message = str(exc)
        finally:
            async with self:
                self.is_running = False
