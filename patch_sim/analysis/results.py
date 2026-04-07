"""Data structures for analysis results.

Defines dataclasses returned by analysis functions in this subpackage.
All classes are plain dataclasses (no Reflex dependency) and are fully
serializable via :func:`dataclasses.asdict`.
"""

import dataclasses


@dataclasses.dataclass
class SpikeMetrics:
    """Metrics for a single detected action potential.

    Attributes:
        index: Zero-based spike number within the trace.
        threshold_voltage: Membrane voltage at spike onset (mV), defined as
            the point where dV/dt first exceeds the detection threshold.
        threshold_time: Time of threshold crossing (ms).
        peak_voltage: Maximum membrane voltage during the spike (mV).
        peak_time: Time of peak voltage (ms).
        rise_time: Duration from threshold crossing to peak (ms).
        half_width: Spike duration measured at the half-amplitude level
            between threshold voltage and peak voltage (ms).
        ahp_depth: Minimum membrane voltage after the spike trough (mV).
            ``None`` when no trough region is available (e.g. last spike
            with no subsequent data).
    """

    index: int
    threshold_voltage: float
    threshold_time: float
    peak_voltage: float
    peak_time: float
    rise_time: float
    half_width: float
    ahp_depth: float | None


@dataclasses.dataclass
class IVPoint:
    """Current measurements for a single voltage step in an I-V protocol.

    Attributes:
        voltage_step: Command voltage applied during the step (mV).
        peak_inward_current: Most negative (inward) current during the step
            (µA/cm²).
        peak_outward_current: Most positive (outward) current during the step
            (µA/cm²).
        steady_state_current: Mean current over the last 10% of the stimulus
            window (µA/cm²).
    """

    voltage_step: float
    peak_inward_current: float
    peak_outward_current: float
    steady_state_current: float


@dataclasses.dataclass
class IVAnalysisResult:
    """Complete I-V analysis from a multi-sweep voltage clamp simulation.

    Points are stored in ascending order of voltage step.  The convenience
    properties ``voltage_steps``, ``peak_inward_currents``,
    ``peak_outward_currents``, and ``steady_state_currents`` extract the
    corresponding field from each point on demand.

    Attributes:
        points: Per-step measurements, one entry per voltage step, sorted by
            ascending voltage.
    """

    points: list[IVPoint]

    @property
    def voltage_steps(self) -> list[float]:
        """Command voltages in mV, sorted ascending."""
        return [p.voltage_step for p in self.points]

    @property
    def peak_inward_currents(self) -> list[float]:
        """Most negative current at each step (µA/cm²)."""
        return [p.peak_inward_current for p in self.points]

    @property
    def peak_outward_currents(self) -> list[float]:
        """Most positive current at each step (µA/cm²)."""
        return [p.peak_outward_current for p in self.points]

    @property
    def steady_state_currents(self) -> list[float]:
        """Mean current over the last 10% of each step (µA/cm²)."""
        return [p.steady_state_current for p in self.points]


@dataclasses.dataclass
class APAnalysisResult:
    """Complete action potential analysis of a voltage trace.

    Attributes:
        spike_count: Number of detected spikes.
        spikes: Per-spike metrics for each detected spike.
        isis: Inter-spike intervals in ms (length = spike_count - 1).
        mean_threshold_voltage: Mean threshold voltage across all spikes (mV),
            or ``None`` when no spikes were detected.
        mean_peak_voltage: Mean peak voltage across all spikes (mV), or
            ``None`` when no spikes were detected.
        mean_rise_time: Mean rise time across all spikes (ms), or ``None``
            when no spikes were detected.
        mean_half_width: Mean half-width across all spikes (ms), or ``None``
            when no spikes were detected.
        mean_ahp_depth: Mean AHP depth across spikes that have a measurable
            trough (mV), or ``None`` when no such spikes exist.
        mean_isi: Mean inter-spike interval (ms), or ``None`` when fewer
            than two spikes were detected.
        firing_rate: Mean firing rate in Hz (``1000 / mean_isi``), or
            ``None`` when fewer than two spikes were detected.
    """

    spike_count: int
    spikes: list[SpikeMetrics]
    isis: list[float]
    mean_threshold_voltage: float | None
    mean_peak_voltage: float | None
    mean_rise_time: float | None
    mean_half_width: float | None
    mean_ahp_depth: float | None
    mean_isi: float | None
    firing_rate: float | None
