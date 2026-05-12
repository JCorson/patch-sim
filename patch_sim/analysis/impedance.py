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
    longest_subthreshold_run,
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
#: reference to be considered for a genuine resonance (rather than measurement
#: ripple on an otherwise monotone passive response).  Clearing this margin is
#: necessary but not sufficient — a resonance is only reported when the peak
#: also has a bracketable -3 dB half-power width (see :func:`_quality_factor`).
_RESONANCE_REL_MARGIN: float = 0.05

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
    ``None`` together unless ``|Z(f)|`` has a genuine resonance — an interior
    peak that both rises meaningfully above the low-frequency reference *and*
    has a half-power (−3 dB) width that can be bracketed inside the analysis
    band.  A passive low-pass cell, or one whose mild interior bump has no
    bracketable bandwidth, reports ``None`` for all three.

    Attributes:
        frequencies: Analysis-band frequency axis in Hz, ascending.
        magnitude: ``|Z(f)|`` in kΩ·cm² (``|V̂ / Î|`` where V̂ is in mV and
            Î is in µA/cm²), aligned to ``frequencies``.
        phase: ``∠Z(f)`` in degrees, aligned to ``frequencies``; positive
            values mean the voltage leads the current.
        resonance_frequency: Frequency of the ``|Z|`` resonance peak in Hz, or
            ``None`` when there is no genuine resonance (the maximum lies at a
            band edge, does not clear the prominence margin, or has no
            bracketable −3 dB width).
        peak_impedance: ``|Z|`` at ``resonance_frequency`` in kΩ·cm², or
            ``None`` when there is no genuine resonance.
        quality_factor: Dimensionless ``Q = f_R / FWHM`` where FWHM is the
            full width of the ``|Z|`` peak at the half-power (−3 dB) level, or
            ``None`` when there is no genuine resonance.
        magnitude_mohm: ``|Z(f)|`` converted to absolute MΩ when ``area_cm2``
            is supplied; ``None`` otherwise.
        peak_impedance_mohm: Absolute peak impedance in MΩ when ``area_cm2`` is
            supplied and a genuine resonance exists; ``None`` otherwise.
        area_cm2: Membrane surface area in cm² used for the absolute
            conversion.  ``None`` when only per-area values were requested.
        f_start: Lower edge of the requested analysis band in Hz.
        f_end: Upper edge of the requested analysis band in Hz.
        analyzed_window_ms: ``None`` when the full chirp window was analyzed.
            Otherwise the duration in ms of the spike-free sub-window that was
            analyzed instead (the chirp contained spikes; impedance was
            recovered from the longest contiguous spike-free segment, which
            covers only a sub-band of ``[f_start, f_end]`` — see
            ``frequencies[0]`` / ``frequencies[-1]`` for the actually-covered
            band).
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
    analyzed_window_ms: float | None = None


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


def _impedance_unavailable_reason(
    time: np.ndarray,
    voltage: np.ndarray,
    injected_current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    f_start: float,
    f_end: float,
) -> str | None:
    """Return a human-readable reason why impedance analysis can't proceed.

    Covers the cheap pre-FFT failure modes — shape / NaN / band sanity, a
    too-short chirp window, and a suprathreshold response with no spike-free
    fallback segment long enough for the FFT.  The post-FFT failures (no
    stimulus power in the band, too few in-band bins after trimming) are not
    detected here and must be diagnosed by running :func:`analyze_impedance`
    itself.

    Args:
        time: Time axis in ms.
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.

    Returns:
        ``None`` when the preamble checks pass and the analysis can proceed; a
        single-sentence reason string otherwise.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(injected_current, dtype=float)
    if v.shape != t.shape or i.shape != t.shape:
        return "internal: trace length mismatch."
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(i))):
        return "internal: trace contains non-finite values."
    if f_end <= f_start or f_start < 0.0:
        return f"invalid frequency band [{f_start}, {f_end}] Hz."
    window_ms = stim_end_ms - stim_start_ms
    if window_ms < _MIN_WINDOW_MS:
        return (
            f"the chirp window ({window_ms:.1f} ms) is shorter than the "
            f"minimum {_MIN_WINDOW_MS:.1f} ms needed for a meaningful spectrum."
        )
    mask = (t >= stim_start_ms) & (t < stim_end_ms)
    if int(mask.sum()) < 4:
        return "too few samples in the chirp window."
    t_win, v_win = t[mask], v[mask]
    if is_subthreshold(t_win, v_win):
        return None
    run = longest_subthreshold_run(t_win, v_win)
    if run is None:
        return (
            "the cell fired throughout the chirp — reduce the amplitude or "
            "apply a hyperpolarizing holding current."
        )
    lo, hi = run
    span = float(t_win[hi - 1] - t_win[lo])
    if span < _MIN_WINDOW_MS:
        return (
            f"the longest spike-free segment ({span:.1f} ms) is shorter than the "
            f"minimum {_MIN_WINDOW_MS:.1f} ms — reduce the amplitude or apply a "
            "hyperpolarizing holding current."
        )
    return None


def impedance_unavailable_reason(
    time: np.ndarray,
    voltage: np.ndarray,
    injected_current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    f_start: float,
    f_end: float,
) -> str:
    """Public wrapper around :func:`_impedance_unavailable_reason`.

    Returns the empty string when the preamble checks pass (and the analysis
    would proceed past the windowing / suprathreshold guards), so callers can
    use the result directly as UI copy.  Only the cheap pre-FFT failure modes
    are reported — when this returns ``""`` and :func:`analyze_impedance` still
    returns ``None``, the caller should fall back to a generic
    "too little usable signal in the band" message.

    Args:
        time: Time axis in ms.
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.

    Returns:
        The reason string from :func:`_impedance_unavailable_reason`, or ``""``
        when that function returns ``None``.
    """
    return (
        _impedance_unavailable_reason(
            time,
            voltage,
            injected_current,
            stim_start_ms,
            stim_end_ms,
            f_start,
            f_end,
        )
        or ""
    )


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
    the magnitude and phase spectra plus, when a genuine resonance is present,
    the resonance frequency and quality factor.  A genuine resonance requires
    an interior ``|Z|`` maximum that both clears the prominence margin over the
    low-frequency reference and has a half-power (−3 dB) width bracketable
    inside the band; ``resonance_frequency``, ``peak_impedance`` and
    ``quality_factor`` are otherwise all ``None``.

    Membrane impedance is a linear, small-signal quantity, so spike transients
    are excluded: when the windowed response contains spikes the analysis falls
    back to the longest contiguous spike-free sub-window (a linear chirp's
    frequency is time-dependent, so the recovered profile then covers only a
    sub-band of ``[f_start, f_end]`` — surfaced via
    :attr:`ImpedanceProfile.analyzed_window_ms`).  The analysis bails out
    (returns ``None``) when no such spike-free segment is long enough — see
    :func:`impedance_unavailable_reason` for the user-facing reason.

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
        shorter than :data:`_MIN_WINDOW_MS`, the response fires throughout the
        chirp (no spike-free segment of at least :data:`_MIN_WINDOW_MS`), the
        stimulus has no spectral power in the band, or fewer than
        :data:`_MIN_BAND_BINS` FFT bins remain after band-edge trimming.
    """
    reason = _impedance_unavailable_reason(
        time,
        voltage,
        injected_current,
        stim_start_ms,
        stim_end_ms,
        f_start,
        f_end,
    )
    if reason is not None:
        logger.warning("analyze_impedance: %s", reason)
        return None

    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(injected_current, dtype=float)
    mask = (t >= stim_start_ms) & (t < stim_end_ms)
    t_win, v_win, i_win = t[mask], v[mask], i[mask]

    analyzed_window_ms: float | None
    if is_subthreshold(t_win, v_win):
        analyzed_window_ms = None
    else:
        # _impedance_unavailable_reason already verified that a long-enough
        # spike-free run exists; fall back to it.
        run = longest_subthreshold_run(t_win, v_win)
        if run is None:  # defensive — guarded above
            return None
        lo, hi = run
        t_win, v_win, i_win = t_win[lo:hi], v_win[lo:hi], i_win[lo:hi]
        analyzed_window_ms = float(t_win[-1] - t_win[0])

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
    # A resonance is reported only when the interior |Z| peak both clears the
    # prominence margin and has a measurable -3 dB width, so that f_R, the peak
    # impedance, and Q always populate (or blank) together — a "resonance" with
    # no bracketable half-power bandwidth is just measurement ripple on an
    # otherwise monotone passive response.
    peak_prominent = 0 < peak_idx < mag.size - 1 and float(mag[peak_idx]) > low_ref * (
        1.0 + _RESONANCE_REL_MARGIN
    )
    q = _quality_factor(fb, mag, peak_idx) if peak_prominent else None
    if q is not None:
        resonance_frequency: float | None = float(fb[peak_idx])
        peak_impedance: float | None = float(mag[peak_idx])
        quality_factor: float | None = q
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
        analyzed_window_ms=analyzed_window_ms,
    )
