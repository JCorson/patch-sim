"""Action potential detection and metric extraction.

Provides :func:`analyze_aps` for extracting standard AP metrics from a
voltage trace, and :func:`analyze_aps_from_result` as a convenience wrapper
for :class:`~patch_sim.clamp_simulations.SimulationResult` structured arrays.
"""

# Needed so that the SimulationResult annotation in analyze_aps_from_result is
# not evaluated eagerly at import time — SimulationResult is only imported
# under TYPE_CHECKING to avoid a circular dependency.
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from .results import APAnalysisResult, SpikeMetrics

if TYPE_CHECKING:
    from patch_sim.clamp_simulations import SimulationResult

#: Minimum number of samples that must separate two threshold crossings.
#: Corresponds to a ~1 ms refractory period at the project's standard 40 kHz
#: sampling rate (dt = 0.025 ms).  ``analyze_aps`` accepts arbitrary time
#: arrays, so callers using a different sampling rate should be aware that
#: the effective refractory window will scale with their dt.
_REFRACTORY_SAMPLES: int = 40


def _interpolate_crossing(
    time: np.ndarray,
    voltage: np.ndarray,
    level: float,
    start: int,
    end: int,
    rising: bool,
) -> float:
    """Find the sub-sample time at which voltage crosses a level.

    Searches the window ``time[start:end]`` / ``voltage[start:end]`` for the
    first crossing of ``level`` in the requested direction and returns the
    interpolated crossing time.  If no crossing is found the time of the
    sample closest to ``level`` is returned as a fallback.

    Args:
        time: Time array (ms).
        voltage: Voltage array (mV), same length as ``time``.
        level: Voltage level to find the crossing of (mV).
        start: Start index of the search window (inclusive).
        end: End index of the search window (exclusive).
        rising: If ``True`` search for an upward crossing; if ``False`` a
            downward crossing.

    Returns:
        Interpolated crossing time in ms.
    """
    seg_v = voltage[start:end]
    seg_t = time[start:end]

    if rising:
        above = seg_v >= level
        # First True after a False
        crossings = np.where(~above[:-1] & above[1:])[0]
    else:
        below = seg_v <= level
        crossings = np.where(~below[:-1] & below[1:])[0]

    if len(crossings) == 0:
        # Fallback: time of sample closest to level
        return float(seg_t[np.argmin(np.abs(seg_v - level))])

    i = crossings[0]
    # Linear interpolation between samples i and i+1
    v0, v1 = float(seg_v[i]), float(seg_v[i + 1])
    t0, t1 = float(seg_t[i]), float(seg_t[i + 1])
    frac = (level - v0) / (v1 - v0) if v1 != v0 else 0.0
    return t0 + frac * (t1 - t0)


def analyze_aps(
    time: np.ndarray,
    voltage: np.ndarray,
    dvdt_threshold: float = 20.0,
    min_spike_height: float = -20.0,
) -> APAnalysisResult:
    """Extract action potential metrics from a voltage trace.

    Spikes are detected by finding upward crossings of ``dvdt_threshold`` in
    the dV/dt signal.  For each candidate crossing the subsequent voltage peak
    is located.  Candidates are rejected when the peak is below
    ``min_spike_height`` or when they fall within a refractory window after
    the previous accepted spike.

    Args:
        time: Time array in ms, shape ``(N,)``.
        voltage: Membrane voltage in mV, shape ``(N,)``, same length as
            ``time``.
        dvdt_threshold: dV/dt value (mV/ms) used to identify spike onset.
            The default of 20 mV/ms is standard for HH-type models.
        min_spike_height: Minimum peak voltage (mV) for a candidate event to
            be classified as a spike.  Subthreshold oscillations typically
            peak well below 0 mV.

    Returns:
        An :class:`~patch_sim.analysis.results.APAnalysisResult` containing
        per-spike metrics and aggregate summary statistics.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)

    n = len(time)
    if n < 2:
        return _empty_result()

    dt_arr = np.diff(time)
    dvdt = np.diff(voltage) / dt_arr  # length n-1

    # --- Step 1: upward threshold crossings in dV/dt ---
    above = dvdt >= dvdt_threshold
    # crossing at index i means dvdt[i] is the first sample >= threshold
    crossing_mask = np.zeros(len(dvdt), dtype=bool)
    crossing_mask[1:] = above[1:] & ~above[:-1]
    crossing_mask[0] = above[0]
    crossing_indices = np.where(crossing_mask)[0]

    spikes: list[SpikeMetrics] = []
    last_peak_idx = -_REFRACTORY_SAMPLES - 1

    for c_idx in crossing_indices:
        # Enforce refractory period
        if c_idx - last_peak_idx < _REFRACTORY_SAMPLES:
            continue

        # --- Step 2: find peak after crossing ---
        # Scan forward until dV/dt drops below zero (voltage starts falling)
        search_start = c_idx + 1
        drop_indices = np.where(dvdt[search_start:] < 0)[0]
        if len(drop_indices) == 0:
            # Voltage never falls — peak is the last sample
            peak_idx = n - 1
        else:
            peak_idx = search_start + drop_indices[0]
            # The actual peak is the maximum in [c_idx, peak_idx]
            peak_idx = int(np.argmax(voltage[c_idx : peak_idx + 1])) + c_idx

        # Reject subthreshold events
        if voltage[peak_idx] < min_spike_height:
            continue

        # --- Step 3: per-spike metrics ---
        thresh_v = float(voltage[c_idx])
        thresh_t = float(time[c_idx])
        peak_v = float(voltage[peak_idx])
        peak_t = float(time[peak_idx])
        rise_time = peak_t - thresh_t

        # Half-width: duration at half-amplitude level
        half_level = (thresh_v + peak_v) / 2.0
        rise_cross_t = _interpolate_crossing(
            time, voltage, half_level, c_idx, peak_idx + 1, rising=True
        )
        # Search for falling crossing after the peak
        fall_end = n
        fall_cross_t = _interpolate_crossing(
            time, voltage, half_level, peak_idx, fall_end, rising=False
        )
        half_width = fall_cross_t - rise_cross_t
        if half_width < 0:
            raise ValueError(
                f"Spike {len(spikes)}: computed half_width={half_width:.4f} ms is "
                "negative (rise_cross_t={rise_cross_t:.4f}, "
                f"fall_cross_t={fall_cross_t:.4f}).  This indicates a bug in the "
                "crossing-detection logic."
            )

        spikes.append(
            SpikeMetrics(
                index=len(spikes),
                threshold_voltage=thresh_v,
                threshold_time=thresh_t,
                peak_voltage=peak_v,
                peak_time=peak_t,
                rise_time=rise_time,
                half_width=half_width,
                ahp_depth=None,  # filled in below once all peaks are known
            )
        )
        last_peak_idx = peak_idx

    if not spikes:
        return _empty_result()

    # --- Step 4: AHP depth ---
    # Collect peak indices again from the SpikeMetrics for the AHP window.
    peak_times = [s.peak_time for s in spikes]
    thresh_times = [s.threshold_time for s in spikes]

    for i, spike in enumerate(spikes):
        p_t = peak_times[i]
        # AHP window: from peak to the next threshold crossing (or end of trace)
        ahp_end_t = thresh_times[i + 1] if i + 1 < len(spikes) else float(time[-1])
        p_idx = int(np.searchsorted(time, p_t))
        ahp_end_idx = int(np.searchsorted(time, ahp_end_t))
        if ahp_end_idx > p_idx:
            spikes[i] = dataclasses.replace(
                spike, ahp_depth=float(np.min(voltage[p_idx : ahp_end_idx + 1]))
            )

    # --- Step 5: ISIs ---
    isis = [peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)]

    # --- Step 6: summary statistics ---
    mean_thresh = float(np.mean([s.threshold_voltage for s in spikes]))
    mean_peak = float(np.mean([s.peak_voltage for s in spikes]))
    mean_rise = float(np.mean([s.rise_time for s in spikes]))
    mean_hw = float(np.mean([s.half_width for s in spikes]))

    ahp_values = [s.ahp_depth for s in spikes if s.ahp_depth is not None]
    mean_ahp: float | None = float(np.mean(ahp_values)) if ahp_values else None

    mean_isi: float | None = float(np.mean(isis)) if isis else None
    firing_rate: float | None = (1000.0 / mean_isi) if mean_isi is not None else None

    return APAnalysisResult(
        spike_count=len(spikes),
        spikes=spikes,
        isis=isis,
        mean_threshold_voltage=mean_thresh,
        mean_peak_voltage=mean_peak,
        mean_rise_time=mean_rise,
        mean_half_width=mean_hw,
        mean_ahp_depth=mean_ahp,
        mean_isi=mean_isi,
        firing_rate=firing_rate,
    )


def analyze_aps_from_result(
    result: SimulationResult,
    dvdt_threshold: float = 20.0,
    min_spike_height: float = -20.0,
) -> APAnalysisResult:
    """Extract AP metrics from a :class:`~patch_sim.clamp_simulations.SimulationResult`.

    Convenience wrapper around :func:`analyze_aps` that pulls the ``"time"``
    and ``"voltage"`` fields from the structured array.

    Args:
        result: A structured NumPy array returned by
            :func:`~patch_sim.simulate_current_clamp` or the voltage-clamp
            equivalents.  Must contain ``"time"`` and ``"voltage"`` fields.
        dvdt_threshold: dV/dt threshold for spike onset detection (mV/ms).
        min_spike_height: Minimum peak voltage for a valid spike (mV).

    Returns:
        An :class:`~patch_sim.analysis.results.APAnalysisResult`.
    """
    return analyze_aps(
        result["time"],
        result["voltage"],
        dvdt_threshold=dvdt_threshold,
        min_spike_height=min_spike_height,
    )


def _empty_result() -> APAnalysisResult:
    """Return an APAnalysisResult representing a trace with no detected spikes.

    Returns:
        An :class:`~patch_sim.analysis.results.APAnalysisResult` with
        ``spike_count`` of 0 and all summary fields set to ``None``.
    """
    return APAnalysisResult(
        spike_count=0,
        spikes=[],
        isis=[],
        mean_threshold_voltage=None,
        mean_peak_voltage=None,
        mean_rise_time=None,
        mean_half_width=None,
        mean_ahp_depth=None,
        mean_isi=None,
        firing_rate=None,
    )
