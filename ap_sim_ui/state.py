"""Application state for the ap_sim web UI.

All reactive variables and event handlers live here. The state drives
the Reflex component tree via computed properties.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import reflex as rx

import ap_sim
from ap_sim_ui import constants, presets


class Sweep(rx.Base):
    """A saved simulation result for sweep overlay display."""

    label: str
    color: str
    # Serialised as list[list[float]] — [time, ...] and [values, ...]
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
    g_Na: float = constants.DEFAULT_G_NA
    g_K: float = constants.DEFAULT_G_K
    g_L: float = constants.DEFAULT_G_L
    C_m: float = constants.DEFAULT_C_M
    v_rest: float = constants.DEFAULT_V_REST
    Na_out: float = constants.DEFAULT_NA_OUT
    Na_in: float = constants.DEFAULT_NA_IN
    K_out: float = constants.DEFAULT_K_OUT
    K_in: float = constants.DEFAULT_K_IN
    Cl_out: float = constants.DEFAULT_CL_OUT
    Cl_in: float = constants.DEFAULT_CL_IN
    T: float = constants.DEFAULT_T

    # ------------------------------------------------------------------ #
    # Experiment mode                                                     #
    # ------------------------------------------------------------------ #
    clamp_mode: str = "Current Clamp"  # "Current Clamp" | "Voltage Clamp"
    sampling_frequency: float = constants.DEFAULT_SAMPLING_FREQUENCY

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
    def figure_data(self) -> dict:
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
            fig.update_yaxes(
                title_text="Current (µA/cm²)", row=stimulus_row, col=1
            )
        else:
            fig.update_yaxes(title_text="Current (µA/cm²)", row=1, col=1)
            fig.update_yaxes(title_text="Voltage (mV)", row=stimulus_row, col=1)

        if show_gating and gating_row is not None:
            fig.update_yaxes(
                title_text="Gating", row=gating_row, col=1, range=[0, 1]
            )

        fig.update_xaxes(title_text="Time (ms)", row=stimulus_row, col=1)
        fig.update_layout(
            height=500,
            margin={"l": 60, "r": 20, "t": 30, "b": 40},
            legend={"orientation": "h", "y": 1.08},
            template="plotly_white",
            hovermode="x unified",
        )
        return fig.to_dict()

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

            fs = self.sampling_frequency
            mode = self.clamp_mode
            ptype = self.protocol_type

            if mode == "Current Clamp":
                stimulus = self._build_current_protocol(fs, ptype)
                df = ap_sim.simulate_current_clamp(neuron, stimulus, fs)
            else:
                stimulus = self._build_voltage_protocol(fs, ptype)
                df = ap_sim.simulate_voltage_clamp(neuron, stimulus, fs)

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
                    self.result_sodium_inactivation = df[
                        "sodium_inactivation"
                    ].tolist()
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
                    self.result_sodium_inactivation = df[
                        "sodium_inactivation"
                    ].tolist()

        except ValueError as exc:
            async with self:
                self.error_message = str(exc)
        finally:
            async with self:
                self.is_running = False

    def _build_current_protocol(
        self, fs: float, ptype: str
    ) -> "np.ndarray":
        """Build a current protocol array from current state variables.

        Args:
            fs: Sampling frequency in Hz.
            ptype: Protocol type name.

        Returns:
            Current protocol array in uA/cm^2.
        """
        if ptype == "Step":
            return ap_sim.step_current(
                duration=self.duration,
                current_amplitude=self.current_amplitude,
                step_start=self.step_start,
                step_duration=self.step_duration,
                sampling_frequency=fs,
            )
        elif ptype == "Ramp":
            return ap_sim.ramp_current(
                duration=self.duration,
                start_current=self.start_current,
                end_current=self.end_current,
                ramp_start=self.ramp_start,
                ramp_duration=self.ramp_duration,
                sampling_frequency=fs,
            )
        elif ptype == "Pulse Train":
            return ap_sim.pulse_train(
                duration=self.duration,
                pulse_amplitude=self.pulse_amplitude,
                pulse_width=self.pulse_width,
                pulse_interval=self.pulse_interval,
                train_start=self.train_start,
                sampling_frequency=fs,
            )
        elif ptype == "Sinusoidal":
            return ap_sim.sinusoidal_current(
                duration=self.duration,
                dc_offset=self.dc_offset,
                amplitude=self.amplitude,
                frequency=self.frequency,
                sampling_frequency=fs,
            )
        elif ptype == "Chirp":
            return ap_sim.chirp_current(
                duration=self.duration,
                dc_offset=self.dc_offset,
                amplitude=self.amplitude,
                start_frequency=self.start_frequency,
                end_frequency=self.end_frequency,
                sampling_frequency=fs,
            )
        else:  # Noise
            return ap_sim.noise_current(
                duration=self.duration,
                mean_current=self.mean_current,
                std_current=self.std_current,
                sampling_frequency=fs,
            )

    def _build_voltage_protocol(
        self, fs: float, ptype: str
    ) -> "np.ndarray":
        """Build a voltage protocol array from current state variables.

        Args:
            fs: Sampling frequency in Hz.
            ptype: Protocol type name.

        Returns:
            Voltage protocol array in mV.
        """
        if ptype == "Step":
            return ap_sim.step_voltage(
                duration=self.duration,
                voltage_amplitude=self.vc_voltage_amplitude,
                step_start=self.vc_step_start,
                step_duration=self.vc_step_duration,
                holding_voltage=self.vc_holding_voltage,
                sampling_frequency=fs,
            )
        elif ptype == "Ramp":
            return ap_sim.ramp_voltage(
                duration=self.duration,
                start_voltage=self.vc_start_voltage,
                end_voltage=self.vc_end_voltage,
                ramp_start=self.vc_ramp_start,
                ramp_duration=self.vc_ramp_duration,
                holding_voltage=self.vc_holding_voltage,
                sampling_frequency=fs,
            )
        elif ptype == "Pulse Train":
            return ap_sim.pulse_train_voltage(
                duration=self.duration,
                pulse_amplitude=self.vc_pulse_amplitude,
                pulse_width=self.vc_pulse_width,
                pulse_interval=self.vc_pulse_interval,
                train_start=self.vc_train_start,
                holding_voltage=self.vc_holding_voltage,
                sampling_frequency=fs,
            )
        elif ptype == "I-V Curve":
            return ap_sim.iv_curve_protocol(
                step_duration=self.duration,
                voltage_min=self.vc_voltage_min,
                voltage_max=self.vc_voltage_max,
                voltage_step=self.vc_voltage_step,
                pre_pulse_duration=self.vc_pre_pulse_duration,
                post_pulse_duration=self.vc_post_pulse_duration,
                holding_voltage=self.vc_holding_voltage,
                sampling_frequency=fs,
            )
        else:  # Activation
            return ap_sim.activation_protocol(
                test_duration=self.duration,
                prepulse_voltage=self.vc_prepulse_voltage,
                prepulse_duration=self.vc_prepulse_duration,
                test_voltage_min=self.vc_test_voltage_min,
                test_voltage_max=self.vc_test_voltage_max,
                voltage_step=self.vc_voltage_step,
                interpulse_duration=self.vc_interpulse_duration,
                holding_voltage=self.vc_holding_voltage,
                sampling_frequency=fs,
            )
