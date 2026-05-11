"""Sweep data model for simulation result transfer and overlay display."""

import numpy as np
from pydantic import BaseModel

from patch_sim.analysis.derivatives import compute_dvdt

# Top-level columns that are not stored in either per-channel dict: time,
# voltage, and the summed total current.  Every other column is sorted into
# ``channel_currents`` (currents) or ``channel_gating`` (gating variables).
_RESERVED_COLUMNS = frozenset({"time", "voltage", "Itotal"})


class Sweep(BaseModel):
    """A simulation result snapshot used for overlay display.

    Attributes:
        label: Display name shown in the plot legend.
        color: Hex color string; empty string means Plotly chooses the color.
        time: Time axis values in ms.
        voltage: Membrane voltage in mV.
        dvdt: Time derivative of membrane voltage in mV/ms.
        total_current: Sum of ion currents in µA/cm².
        stimulus: Stimulus waveform (current µA/cm² or voltage command mV).
        clamp_mode: CURRENT_CLAMP or VOLTAGE_CLAMP.
        channel_currents: Per-channel currents keyed by simulation column
            name (``"INa"``, ``"IK"``, ``"INaL"``, ``"IKL"``, ``"Ih"``,
            ``"IKv31"``, …).  Channels absent from a preset are omitted.
        channel_gating: Per-channel gating variable traces keyed by variable
            name (``"n"``, ``"m"``, ``"h"``, ``"sNa12"``, ``"r"``, ``"a"``,
            …).  Gates absent from a preset are omitted.
    """

    label: str
    color: str
    time: list[float]
    voltage: list[float]
    dvdt: list[float]
    total_current: list[float]
    stimulus: list[float]
    clamp_mode: str
    channel_currents: dict[str, list[float]] = {}
    channel_gating: dict[str, list[float]] = {}

    @classmethod
    def from_result(
        cls,
        result,
        stimulus,
        label: str,
        color: str,
        mode: str,
    ) -> "Sweep":
        """Create a Sweep from a simulation result structured array.

        Per-channel current columns (anything starting with ``"I"`` other
        than ``"Itotal"``) become entries in ``channel_currents``; remaining
        non-reserved columns become ``channel_gating``.

        Args:
            result: Simulation result structured array with a ``"time"`` field.
            stimulus: Stimulus array (current or voltage command).
            label: Display name for this sweep in the legend.
            color: Hex color string; pass empty string to use Plotly default.
            mode: Clamp mode — CURRENT_CLAMP or VOLTAGE_CLAMP.

        Returns:
            A fully populated Sweep instance.
        """
        columns = list(result.dtype.names)

        def _col(name: str) -> list[float]:
            """Return field as list, or empty list if field absent."""
            return result[name].tolist() if name in columns else []

        channel_currents: dict[str, list[float]] = {}
        channel_gating: dict[str, list[float]] = {}
        for col in columns:
            if col in _RESERVED_COLUMNS:
                continue
            if col.startswith("I"):
                channel_currents[col] = result[col].tolist()
            else:
                channel_gating[col] = result[col].tolist()

        time_arr = result["time"]
        if "voltage" in columns:
            dvdt_arr = compute_dvdt(time_arr, result["voltage"])
        else:
            dvdt_arr = np.array([])

        return cls(
            label=label,
            color=color,
            clamp_mode=mode,
            time=time_arr.tolist(),
            stimulus=stimulus.tolist(),
            voltage=_col("voltage"),
            dvdt=dvdt_arr.tolist(),
            total_current=_col("Itotal"),
            channel_currents=channel_currents,
            channel_gating=channel_gating,
        )
