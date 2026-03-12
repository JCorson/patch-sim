"""Application state for the ap_sim web UI.

All reactive variables and event handlers live here. The state drives
the Reflex component tree via computed properties.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
import reflex as rx

import ap_sim
import ap_sim.clamp_simulations
from ap_sim.constants import (
    DEFAULT_C_M,
    DEFAULT_G_K,
    DEFAULT_G_L,
    DEFAULT_G_NA,
    DEFAULT_K_IN,
    DEFAULT_K_OUT,
    DEFAULT_NA_IN,
    DEFAULT_NA_OUT,
    DEFAULT_T,
    DEFAULT_V_REST,
    DEFAULT_CL_IN,
    DEFAULT_CL_OUT,
)
from ap_sim_ui import constants, presets
from ap_sim_ui.protocol_builders import build_current_protocol, build_voltage_protocol


class Sweep(BaseModel):
    """A saved simulation result for sweep overlay display."""

    label: str
    color: str
    # Serialized as list[list[float]] — [time, ...] and [values, ...]
    time: list[float]
    voltage: list[float]
    sodium_current: list[float]
    potassium_current: list[float]
    leak_current: list[float]
    total_current: list[float]
    potassium_activation: list[float]
    sodium_activation: list[float]
    sodium_inactivation: list[float]
    stimulus: list[float]
    clamp_mode: str  # "Current Clamp" or "Voltage Clamp"


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
    # Simulation results (current sweep)                                 #
    # ------------------------------------------------------------------ #
    result_time: list[float] = []
    result_voltage: list[float] = []
    result_sodium_current: list[float] = []
    result_potassium_current: list[float] = []
    result_leak_current: list[float] = []
    result_total_current: list[float] = []
    result_potassium_activation: list[float] = []
    result_sodium_activation: list[float] = []
    result_sodium_inactivation: list[float] = []
    result_stimulus: list[float] = []
    result_clamp_mode: str = ""

    # ------------------------------------------------------------------ #
    # Sweep overlay                                                       #
    # ------------------------------------------------------------------ #
    sweeps: list[Sweep] = []

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
        return len(self.result_time) > 0

    @rx.var
    def figure_data(self) -> go.Figure:
        """Plotly figure dict rebuilt whenever relevant state changes."""
        mode = self.result_clamp_mode or self.clamp_mode
        time = self.result_time

        show_gating = (
            self.show_potassium_activation
            or self.show_sodium_activation
            or self.show_sodium_inactivation
        )
        rows = 3 if show_gating else 2
        row_heights = [0.5, 0.25, 0.25] if show_gating else [0.6, 0.4]
        stimulus_row = 3 if show_gating else 2
        gating_row = 2 if show_gating else None

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.08,
        )

        def _scatter(x, y, name, row, color=None):
            """Add a scattergl trace."""
            line = {"color": color} if color else {}
            fig.add_trace(
                go.Scattergl(x=x, y=y, name=name, mode="lines", line=line),
                row=row,
                col=1,
            )

        if self.has_result:
            if mode == "Current Clamp":
                if self.show_voltage:
                    _scatter(time, self.result_voltage, "Voltage (mV)", 1)
            else:
                if self.show_total_current:
                    _scatter(time, self.result_total_current, "Total I", 1)
                if self.show_sodium_current:
                    _scatter(time, self.result_sodium_current, "I_Na", 1)
                if self.show_potassium_current:
                    _scatter(time, self.result_potassium_current, "I_K", 1)
                if self.show_leak_current:
                    _scatter(time, self.result_leak_current, "I_L", 1)

            if show_gating and gating_row is not None:
                if self.show_potassium_activation:
                    _scatter(time, self.result_potassium_activation, "n", gating_row)
                if self.show_sodium_activation:
                    _scatter(time, self.result_sodium_activation, "m", gating_row)
                if self.show_sodium_inactivation:
                    _scatter(time, self.result_sodium_inactivation, "h", gating_row)

            stim_label = (
                "Stimulus (µA/cm²)" if mode == "Current Clamp" else "Command (mV)"
            )
            _scatter(time, self.result_stimulus, stim_label, stimulus_row)

        for sweep in self.sweeps:
            c = sweep.color
            if sweep.clamp_mode == "Current Clamp":
                _scatter(sweep.time, sweep.voltage, f"{sweep.label} V", 1, c)
            else:
                _scatter(
                    sweep.time, sweep.total_current, f"{sweep.label} I_total", 1, c
                )

        if mode == "Current Clamp":
            fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
            fig.update_yaxes(title_text="Current (µA/cm²)", row=stimulus_row, col=1)
        else:
            fig.update_yaxes(title_text="Current (µA/cm²)", row=1, col=1)
            fig.update_yaxes(title_text="Voltage (mV)", row=stimulus_row, col=1)

        if show_gating and gating_row is not None:
            fig.update_yaxes(title_text="Gating", row=gating_row, col=1, range=[0, 1])

        fig.update_xaxes(title_text="Time (ms)", row=stimulus_row, col=1)
        fig.update_layout(
            height=500,
            margin={"l": 60, "r": 20, "t": 30, "b": 40},
            legend={"orientation": "h", "y": 1.08},
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
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
    # Reflex's auto-generated set_X handlers are typed `value: float`,   #
    # but rx.input.on_change passes str and rx.slider.on_change passes   #
    # list[float].  These explicit thin wrappers accept both.            #

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

    # Neuron parameters
    def set_g_Na(self, value: "str | list[float]") -> None:
        """Set g_Na from an input or slider event."""
        self._set_float("g_Na", value)

    def set_g_K(self, value: "str | list[float]") -> None:
        """Set g_K from an input or slider event."""
        self._set_float("g_K", value)

    def set_g_L(self, value: "str | list[float]") -> None:
        """Set g_L from an input or slider event."""
        self._set_float("g_L", value)

    def set_C_m(self, value: "str | list[float]") -> None:
        """Set C_m from an input or slider event."""
        self._set_float("C_m", value)

    def set_v_rest(self, value: "str | list[float]") -> None:
        """Set v_rest from an input or slider event."""
        self._set_float("v_rest", value)

    def set_Na_out(self, value: "str | list[float]") -> None:
        """Set Na_out from an input or slider event."""
        self._set_float("Na_out", value)

    def set_Na_in(self, value: "str | list[float]") -> None:
        """Set Na_in from an input or slider event."""
        self._set_float("Na_in", value)

    def set_K_out(self, value: "str | list[float]") -> None:
        """Set K_out from an input or slider event."""
        self._set_float("K_out", value)

    def set_K_in(self, value: "str | list[float]") -> None:
        """Set K_in from an input or slider event."""
        self._set_float("K_in", value)

    def set_Cl_out(self, value: "str | list[float]") -> None:
        """Set Cl_out from an input or slider event."""
        self._set_float("Cl_out", value)

    def set_Cl_in(self, value: "str | list[float]") -> None:
        """Set Cl_in from an input or slider event."""
        self._set_float("Cl_in", value)

    def set_T(self, value: "str | list[float]") -> None:
        """Set T from an input or slider event."""
        self._set_float("T", value)

    # Shared protocol params
    def set_duration(self, value: "str | list[float]") -> None:
        """Set duration from an input or slider event."""
        self._set_float("duration", value)

    # Current clamp protocol params
    def set_current_amplitude(self, value: "str | list[float]") -> None:
        """Set current_amplitude from an input or slider event."""
        self._set_float("current_amplitude", value)

    def set_step_start(self, value: "str | list[float]") -> None:
        """Set step_start from an input or slider event."""
        self._set_float("step_start", value)

    def set_step_duration(self, value: "str | list[float]") -> None:
        """Set step_duration from an input or slider event."""
        self._set_float("step_duration", value)

    def set_start_current(self, value: "str | list[float]") -> None:
        """Set start_current from an input or slider event."""
        self._set_float("start_current", value)

    def set_end_current(self, value: "str | list[float]") -> None:
        """Set end_current from an input or slider event."""
        self._set_float("end_current", value)

    def set_ramp_start(self, value: "str | list[float]") -> None:
        """Set ramp_start from an input or slider event."""
        self._set_float("ramp_start", value)

    def set_ramp_duration(self, value: "str | list[float]") -> None:
        """Set ramp_duration from an input or slider event."""
        self._set_float("ramp_duration", value)

    def set_pulse_amplitude(self, value: "str | list[float]") -> None:
        """Set pulse_amplitude from an input or slider event."""
        self._set_float("pulse_amplitude", value)

    def set_pulse_width(self, value: "str | list[float]") -> None:
        """Set pulse_width from an input or slider event."""
        self._set_float("pulse_width", value)

    def set_pulse_interval(self, value: "str | list[float]") -> None:
        """Set pulse_interval from an input or slider event."""
        self._set_float("pulse_interval", value)

    def set_train_start(self, value: "str | list[float]") -> None:
        """Set train_start from an input or slider event."""
        self._set_float("train_start", value)

    def set_dc_offset(self, value: "str | list[float]") -> None:
        """Set dc_offset from an input or slider event."""
        self._set_float("dc_offset", value)

    def set_amplitude(self, value: "str | list[float]") -> None:
        """Set amplitude from an input or slider event."""
        self._set_float("amplitude", value)

    def set_frequency(self, value: "str | list[float]") -> None:
        """Set frequency from an input or slider event."""
        self._set_float("frequency", value)

    def set_start_frequency(self, value: "str | list[float]") -> None:
        """Set start_frequency from an input or slider event."""
        self._set_float("start_frequency", value)

    def set_end_frequency(self, value: "str | list[float]") -> None:
        """Set end_frequency from an input or slider event."""
        self._set_float("end_frequency", value)

    def set_mean_current(self, value: "str | list[float]") -> None:
        """Set mean_current from an input or slider event."""
        self._set_float("mean_current", value)

    def set_std_current(self, value: "str | list[float]") -> None:
        """Set std_current from an input or slider event."""
        self._set_float("std_current", value)

    # Voltage clamp protocol params
    def set_vc_voltage_amplitude(self, value: "str | list[float]") -> None:
        """Set vc_voltage_amplitude from an input or slider event."""
        self._set_float("vc_voltage_amplitude", value)

    def set_vc_step_start(self, value: "str | list[float]") -> None:
        """Set vc_step_start from an input or slider event."""
        self._set_float("vc_step_start", value)

    def set_vc_step_duration(self, value: "str | list[float]") -> None:
        """Set vc_step_duration from an input or slider event."""
        self._set_float("vc_step_duration", value)

    def set_vc_holding_voltage(self, value: "str | list[float]") -> None:
        """Set vc_holding_voltage from an input or slider event."""
        self._set_float("vc_holding_voltage", value)

    def set_vc_start_voltage(self, value: "str | list[float]") -> None:
        """Set vc_start_voltage from an input or slider event."""
        self._set_float("vc_start_voltage", value)

    def set_vc_end_voltage(self, value: "str | list[float]") -> None:
        """Set vc_end_voltage from an input or slider event."""
        self._set_float("vc_end_voltage", value)

    def set_vc_ramp_start(self, value: "str | list[float]") -> None:
        """Set vc_ramp_start from an input or slider event."""
        self._set_float("vc_ramp_start", value)

    def set_vc_ramp_duration(self, value: "str | list[float]") -> None:
        """Set vc_ramp_duration from an input or slider event."""
        self._set_float("vc_ramp_duration", value)

    def set_vc_pulse_amplitude(self, value: "str | list[float]") -> None:
        """Set vc_pulse_amplitude from an input or slider event."""
        self._set_float("vc_pulse_amplitude", value)

    def set_vc_pulse_width(self, value: "str | list[float]") -> None:
        """Set vc_pulse_width from an input or slider event."""
        self._set_float("vc_pulse_width", value)

    def set_vc_pulse_interval(self, value: "str | list[float]") -> None:
        """Set vc_pulse_interval from an input or slider event."""
        self._set_float("vc_pulse_interval", value)

    def set_vc_train_start(self, value: "str | list[float]") -> None:
        """Set vc_train_start from an input or slider event."""
        self._set_float("vc_train_start", value)

    def set_vc_voltage_min(self, value: "str | list[float]") -> None:
        """Set vc_voltage_min from an input or slider event."""
        self._set_float("vc_voltage_min", value)

    def set_vc_voltage_max(self, value: "str | list[float]") -> None:
        """Set vc_voltage_max from an input or slider event."""
        self._set_float("vc_voltage_max", value)

    def set_vc_voltage_step(self, value: "str | list[float]") -> None:
        """Set vc_voltage_step from an input or slider event."""
        self._set_float("vc_voltage_step", value)

    def set_vc_pre_pulse_duration(self, value: "str | list[float]") -> None:
        """Set vc_pre_pulse_duration from an input or slider event."""
        self._set_float("vc_pre_pulse_duration", value)

    def set_vc_post_pulse_duration(self, value: "str | list[float]") -> None:
        """Set vc_post_pulse_duration from an input or slider event."""
        self._set_float("vc_post_pulse_duration", value)

    def set_vc_prepulse_voltage(self, value: "str | list[float]") -> None:
        """Set vc_prepulse_voltage from an input or slider event."""
        self._set_float("vc_prepulse_voltage", value)

    def set_vc_prepulse_duration(self, value: "str | list[float]") -> None:
        """Set vc_prepulse_duration from an input or slider event."""
        self._set_float("vc_prepulse_duration", value)

    def set_vc_test_voltage_min(self, value: "str | list[float]") -> None:
        """Set vc_test_voltage_min from an input or slider event."""
        self._set_float("vc_test_voltage_min", value)

    def set_vc_test_voltage_max(self, value: "str | list[float]") -> None:
        """Set vc_test_voltage_max from an input or slider event."""
        self._set_float("vc_test_voltage_max", value)

    def set_vc_interpulse_duration(self, value: "str | list[float]") -> None:
        """Set vc_interpulse_duration from an input or slider event."""
        self._set_float("vc_interpulse_duration", value)

    def add_sweep(self) -> None:
        """Save the current simulation result as a named sweep."""
        if not self.has_result:
            return
        sweep_index = len(self.sweeps)
        color = constants.SWEEP_COLORS[sweep_index % len(constants.SWEEP_COLORS)]
        self.sweeps.append(
            Sweep(
                label=f"Sweep {sweep_index + 1}",
                color=color,
                time=self.result_time,
                voltage=self.result_voltage,
                sodium_current=self.result_sodium_current,
                potassium_current=self.result_potassium_current,
                leak_current=self.result_leak_current,
                total_current=self.result_total_current,
                potassium_activation=self.result_potassium_activation,
                sodium_activation=self.result_sodium_activation,
                sodium_inactivation=self.result_sodium_inactivation,
                stimulus=self.result_stimulus,
                clamp_mode=self.result_clamp_mode,
            )
        )

    def clear_sweeps(self) -> None:
        """Remove all saved sweeps."""
        self.sweeps = []

    @rx.event(background=True)
    async def run_simulation(self) -> None:
        """Build protocol and run the simulation asynchronously.

        Uses background=True so the UI stays responsive during long runs.
        """
        async with self:
            self.is_running = True
            self.error_message = ""

        try:
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

            time_vals = df.index.tolist()
            stim_list = stimulus.tolist()

            async with self:
                self.result_time = time_vals
                self.result_stimulus = stim_list
                self.result_clamp_mode = mode
                if mode == "Current Clamp":
                    self.result_voltage = df["voltage"].tolist()
                    self.result_potassium_activation = df[
                        "potassium_activation"
                    ].tolist()
                    self.result_sodium_activation = df["sodium_activation"].tolist()
                    self.result_sodium_inactivation = df["sodium_inactivation"].tolist()
                    self.result_total_current = []
                    self.result_sodium_current = []
                    self.result_potassium_current = []
                    self.result_leak_current = []
                else:
                    self.result_voltage = df["voltage"].tolist()
                    self.result_total_current = df["total_current"].tolist()
                    self.result_sodium_current = df["sodium_current"].tolist()
                    self.result_potassium_current = df["potassium_current"].tolist()
                    self.result_leak_current = df["leak_current"].tolist()
                    self.result_potassium_activation = df[
                        "potassium_activation"
                    ].tolist()
                    self.result_sodium_activation = df["sodium_activation"].tolist()
                    self.result_sodium_inactivation = df["sodium_inactivation"].tolist()

        except ValueError as exc:
            async with self:
                self.error_message = str(exc)
        finally:
            async with self:
                self.is_running = False
