"""Membrane impedance profile from a chirp current-clamp run.

Provides :func:`analyze_impedance` for computing the FFT-based membrane
impedance ``Z(f) = V̂(f) / Î(f)`` over the swept frequency band of a chirp
(linear frequency-sweep) current-clamp stimulus.  Neurons with Ih/HCN
channels show subthreshold resonance — an interior peak in ``|Z(f)|`` —
which this analysis quantifies via the resonance frequency ``f_R`` and the
quality factor ``Q``.

Data classes:
    ImpedanceProfile: Impedance magnitude/phase spectra and resonance metrics.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from patch_sim.analysis.passive_properties import (
    density_to_absolute_r_in,
    is_subthreshold,
)

logger = logging.getLogger(__name__)

#: Minimum chirp-window duration (ms) required for a meaningful spectrum.  A
#: shorter window gives a frequency resolution too coarse to resolve a
#: subthreshold resonance peak.
_MIN_WINDOW_MS: float = 50.0

#: Minimum number of FFT bins that must fall inside the analysis band for the
#: result to be considered meaningful.
_MIN_BAND_BINS: int = 8

#: Fractional margin by which the ``|Z|`` peak must exceed the low-frequency
#: reference to count as a genuine resonance (rather than measurement ripple on
#: an otherwise monotone passive response).
_RESONANCE_REL_MARGIN: float = 0.01

#: A linear chirp's spectral power tapers toward the band edges.  Contiguous
#: leading/trailing FFT bins whose stimulus magnitude falls below this fraction
#: of the in-band maximum are trimmed before forming ``Z(f)``, so that dividing
#: by a near-zero stimulus does not inflate the band-edge impedance estimate.
_MIN_STIM_FRAC: float = 0.05


@dataclasses.dataclass
class ImpedanceProfile:
    """Membrane impedance profile from a chirp current-clamp run.

    The per-area spectra (``magnitude``) are always populated.  When
    ``area_cm2`` is supplied to :func:`analyze_impedance`, the absolute
    counterparts (``magnitude_mohm``, ``peak_impedance_mohm``) are filled in
    too — analogous to the per-area / absolute split in
    :class:`~patch_sim.analysis.passive_properties.PassiveProperties`.

    ``resonance_frequency``, ``peak_impedance`` and ``quality_factor`` are all
    ``None`` when ``|Z(f)|`` has no interior peak that rises meaningfully above
    the low-frequency reference — the hallmark of a passive low-pass cell with
    no resonance.  ``quality_factor`` may additionally be ``None`` when an
    interior peak exists but its half-power width cannot be bracketed inside
    the analysis band.

    Attributes:
        frequencies: Analysis-band frequency axis in Hz, ascending.
        magnitude: ``|Z(f)|`` in kΩ·cm² (``|V̂ / Î|`` where V̂ is in mV and
            Î is in µA/cm²), aligned to ``frequencies``.
        phase: ``∠Z(f)`` in degrees, aligned to ``frequencies``; positive
            values mean the voltage leads the current.
        resonance_frequency: Frequency of the ``|Z|`` peak in Hz, or ``None``
            when the maximum lies at a band edge or does not rise meaningfully
            above the low-frequency edge (no genuine resonance).
        peak_impedance: ``|Z|`` at ``resonance_frequency`` in kΩ·cm², or
            ``None`` when there is no interior peak.
        quality_factor: Dimensionless ``Q = f_R / FWHM`` where FWHM is the
            full width of the ``|Z|`` peak at the half-power (−3 dB) level, or
            ``None`` when there is no interior peak or the half-power crossings
            cannot be bracketed inside the band.
        magnitude_mohm: ``|Z(f)|`` converted to absolute MΩ when ``area_cm2``
            is supplied; ``None`` otherwise.
        peak_impedance_mohm: Absolute peak impedance in MΩ when ``area_cm2`` is
            supplied and an interior peak exists; ``None`` otherwise.
        area_cm2: Membrane surface area in cm² used for the absolute
            conversion.  ``None`` when only per-area values were requested.
        f_start: Lower edge of the analysis band in Hz.
        f_end: Upper edge of the analysis band in Hz.
    """

    frequencies: list[float]
    magnitude: list[float]
    phase: list[float]
    resonance_frequency: float | None
    peak_impedance: float | None
    quality_factor: float | None
    magnitude_mohm: list[float] | None = None
    peak_impedance_mohm: float | None = None
    area_cm2: float | None = None
    f_start: float = 0.0
    f_end: float = 0.0


def _half_power_crossing(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    peak_idx: int,
    half_power: float,
    direction: int,
) -> float | None:
    """Find the frequency where ``|Z|`` falls to the half-power level.

    Walks outward from ``peak_idx`` in ``direction`` (``-1`` toward lower
    frequencies, ``+1`` toward higher) to the first bin at or below
    ``half_power``, then linearly interpolates the crossing frequency between
    that bin and its inward neighbour.

    Args:
        frequencies: Analysis-band frequency axis in Hz, ascending.
        magnitude: ``|Z(f)|`` aligned to ``frequencies``.
        peak_idx: Index of the ``|Z|`` peak within the band.
        half_power: Magnitude threshold (peak / sqrt(2)).
        direction: ``-1`` to search toward lower bins, ``+1`` toward higher.

    Returns:
        The interpolated crossing frequency in Hz, or ``None`` when the
        threshold is never reached before the band edge.
    """
    idx = peak_idx
    n = magnitude.size
    while 0 <= idx + direction < n:
        nxt = idx + direction
        if magnitude[nxt] <= half_power:
            m0, m1 = float(magnitude[idx]), float(magnitude[nxt])
            f0, f1 = float(frequencies[idx]), float(frequencies[nxt])
            if m0 == m1:
                return f1
            return f0 + (half_power - m0) * (f1 - f0) / (m1 - m0)
        idx = nxt
    return None


def _quality_factor(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    peak_idx: int,
) -> float | None:
    """Estimate the resonance quality factor ``Q = f_R / FWHM``.

    FWHM is the full width of the ``|Z|`` peak at the half-power (−3 dB) level,
    i.e. where ``|Z|`` equals ``peak / sqrt(2)``.  The half-power crossings on
    each side of the peak are found by linear interpolation between bracketing
    FFT bins.

    Args:
        frequencies: Analysis-band frequency axis in Hz, ascending.
        magnitude: ``|Z(f)|`` aligned to ``frequencies``.
        peak_idx: Index of the ``|Z|`` peak within the band.

    Returns:
        The quality factor, or ``None`` when either half-power crossing cannot
        be bracketed inside the band or the bracketed width is non-positive.
    """
    half_power = float(magnitude[peak_idx]) / np.sqrt(2.0)
    f_low = _half_power_crossing(frequencies, magnitude, peak_idx, half_power, -1)
    f_high = _half_power_crossing(frequencies, magnitude, peak_idx, half_power, +1)
    if f_low is None or f_high is None:
        return None
    fwhm = f_high - f_low
    if fwhm <= 0.0:
        return None
    return float(frequencies[peak_idx]) / fwhm


def analyze_impedance(
    time: np.ndarray,
    voltage: np.ndarray,
    injected_current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    f_start: float,
    f_end: float,
    area_cm2: float | None = None,
) -> ImpedanceProfile | None:
    """Compute the membrane impedance profile from a chirp current-clamp run.

    Extracts the chirp window from the traces, removes the DC component from
    both signals, takes the real FFT of each (``numpy.fft.rfft``), forms
    ``Z(f) = V̂(f) / Î(f)`` over the swept band ``[f_start, f_end]`` (trimming
    band-edge bins where the chirp has negligible spectral power), and reports
    the magnitude and phase spectra plus the resonance frequency (argmax of
    ``|Z|``, only when it is an interior maximum that rises above the
    low-frequency reference) and quality factor.

    Membrane impedance is a linear, small-signal quantity: the analysis bails
    out (returns ``None``) when the windowed response is suprathreshold, since
    spike transients dominate the spectrum and the result would be meaningless.

    Args:
        time: Time axis in ms (uniformly sampled).
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.
        area_cm2: Optional membrane surface area in cm².  When supplied, the
            absolute MΩ counterparts (``magnitude_mohm``,
            ``peak_impedance_mohm``) are populated.

    Returns:
        An :class:`ImpedanceProfile`, or ``None`` when the inputs are
        inconsistent (mismatched lengths, non-finite values), the band is
        invalid (``f_end <= f_start`` or ``f_start < 0``), the chirp window is
        shorter than :data:`_MIN_WINDOW_MS`, the windowed response is
        suprathreshold, the stimulus has no spectral power in the band, or
        fewer than :data:`_MIN_BAND_BINS` FFT bins remain after trimming.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(injected_current, dtype=float)
    if v.shape != t.shape or i.shape != t.shape:
        logger.warning("analyze_impedance: trace length mismatch; skipping.")
        return None
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(i))):
        logger.warning("analyze_impedance: non-finite trace values; skipping.")
        return None
    if f_end <= f_start or f_start < 0.0:
        logger.warning(
            "analyze_impedance: invalid band [%s, %s] Hz; skipping.", f_start, f_end
        )
        return None
    if stim_end_ms - stim_start_ms < _MIN_WINDOW_MS:
        logger.warning(
            "analyze_impedance: chirp window %.1f ms shorter than %.1f ms; skipping.",
            stim_end_ms - stim_start_ms,
            _MIN_WINDOW_MS,
        )
        return None

    mask = (t >= stim_start_ms) & (t < stim_end_ms)
    if int(mask.sum()) < 4:
        logger.warning("analyze_impedance: too few samples in chirp window; skipping.")
        return None
    t_win, v_win, i_win = t[mask], v[mask], i[mask]

    if not is_subthreshold(t_win, v_win):
        logger.warning(
            "analyze_impedance: response is suprathreshold; impedance is only "
            "defined in the subthreshold regime; skipping."
        )
        return None

    dt_ms = float(np.mean(np.diff(t_win)))
    if dt_ms <= 0.0:
        logger.warning("analyze_impedance: non-positive sampling interval; skipping.")
        return None
    fs_hz = 1000.0 / dt_ms

    # Remove the DC component from both signals before transforming.  No
    # additional taper is applied: a linear chirp already starts at zero
    # phase, and its magnitude spectrum is approximately flat across the swept
    # band, so a Hann (or similar) window is counter-productive here — it
    # heavily attenuates the chirp's low- and high-frequency content (which
    # lives in its first and last cycles) and inflates the band-edge impedance
    # estimate where the windowed stimulus energy collapses toward zero.
    v_d = v_win - v_win.mean()
    i_d = i_win - i_win.mean()
    n = v_d.size

    v_fft = np.fft.rfft(v_d)
    i_fft = np.fft.rfft(i_d)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    band = (freqs >= f_start) & (freqs <= f_end)
    if int(band.sum()) < _MIN_BAND_BINS:
        logger.warning(
            "analyze_impedance: only %d FFT bins in band; need %d; skipping.",
            int(band.sum()),
            _MIN_BAND_BINS,
        )
        return None
    fb = freqs[band]
    v_band = v_fft[band]
    i_band = i_fft[band]

    i_mag = np.abs(i_band)
    i_max = float(np.max(i_mag))
    if i_max <= 0.0:
        logger.warning("analyze_impedance: stimulus has no power in band; skipping.")
        return None
    # Trim contiguous band-edge bins where the chirp has negligible power so
    # that dividing by a near-zero stimulus does not inflate the estimate.
    keep = np.flatnonzero(i_mag >= _MIN_STIM_FRAC * i_max)
    lo, hi = int(keep[0]), int(keep[-1]) + 1
    if hi - lo < _MIN_BAND_BINS:
        logger.warning(
            "analyze_impedance: only %d FFT bins with usable stimulus power; "
            "need %d; skipping.",
            hi - lo,
            _MIN_BAND_BINS,
        )
        return None
    fb = fb[lo:hi]
    v_band = v_band[lo:hi]
    i_band = i_band[lo:hi]

    z = v_band / i_band
    mag = np.abs(z)
    phase_deg = np.degrees(np.angle(z))

    peak_idx = int(np.argmax(mag))
    # Reference the low-frequency end with a small average to be robust to
    # spectral leakage in the first bin.
    low_ref = float(np.mean(mag[: min(3, mag.size)]))
    is_resonant = 0 < peak_idx < mag.size - 1 and float(mag[peak_idx]) > low_ref * (
        1.0 + _RESONANCE_REL_MARGIN
    )
    if is_resonant:
        resonance_frequency: float | None = float(fb[peak_idx])
        peak_impedance: float | None = float(mag[peak_idx])
        quality_factor: float | None = _quality_factor(fb, mag, peak_idx)
    else:
        resonance_frequency = None
        peak_impedance = None
        quality_factor = None

    if area_cm2 is not None and area_cm2 > 0.0:
        # Per the density_to_absolute_r_in derivation: kΩ·cm² / area / 1000 = MΩ.
        magnitude_mohm: list[float] | None = [float(m) / area_cm2 / 1000.0 for m in mag]
    else:
        magnitude_mohm = None
    peak_impedance_mohm = (
        density_to_absolute_r_in(peak_impedance, area_cm2)
        if peak_impedance is not None
        else None
    )

    return ImpedanceProfile(
        frequencies=fb.tolist(),
        magnitude=mag.tolist(),
        phase=phase_deg.tolist(),
        resonance_frequency=resonance_frequency,
        peak_impedance=peak_impedance,
        quality_factor=quality_factor,
        magnitude_mohm=magnitude_mohm,
        peak_impedance_mohm=peak_impedance_mohm,
        area_cm2=area_cm2,
        f_start=float(f_start),
        f_end=float(f_end),
    )
