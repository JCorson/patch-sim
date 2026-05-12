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

#: A linear chirp's spectral power tapers toward the band edges (and can have
#: occasional interior dropouts from spectral leakage).  After forming the
#: in-band spectrum the analysis keeps only the longest *contiguous* run of FFT
#: bins whose stimulus magnitude is at least this fraction of the in-band
#: maximum, so that dividing by a near-zero stimulus cannot inflate the
#: estimate at the band edges or spike it at an interior dropout.
_MIN_STIM_FRAC: float = 0.05

#: Windowed traces returned by :func:`_prepare_impedance_window` on success:
#: ``(t_win, v_win, i_win, analyzed_window_ms)`` where ``analyzed_window_ms`` is
#: ``None`` for the full chirp window or the sub-window duration in ms.
_PreparedWindow = tuple[np.ndarray, np.ndarray, np.ndarray, float | None]


@dataclasses.dataclass
class ImpedanceProfile:
    """Membrane impedance profile from a chirp current-clamp run.

    The per-area spectra (``magnitude``) and the band-maximum impedance
    (``max_impedance``) are always populated.  When ``area_cm2`` is supplied to
    :func:`analyze_impedance`, the absolute counterparts (``magnitude_mohm``,
    ``max_impedance_mohm``, ``peak_impedance_mohm``) are filled in too —
    analogous to the per-area / absolute split in
    :class:`~patch_sim.analysis.passive_properties.PassiveProperties`.

    ``resonance_frequency``, ``peak_impedance`` and ``quality_factor`` are all
    ``None`` together unless ``|Z(f)|`` has a genuine resonance — an interior
    peak that both rises meaningfully above the low-frequency reference *and*
    has a half-power (−3 dB) width that can be bracketed inside the analysis
    band and is at least one FFT bin wide.  A passive low-pass cell, or one
    whose mild interior bump has no bracketable bandwidth, reports ``None`` for
    all three; its ``max_impedance`` (the band-edge value, ≈ the input
    impedance) is still available.

    Attributes:
        frequencies: Analysis-band frequency axis in Hz, ascending.
        magnitude: ``|Z(f)|`` in kΩ·cm² (``|V̂ / Î|`` where V̂ is in mV and
            Î is in µA/cm²), aligned to ``frequencies``.
        phase: ``∠Z(f)`` in degrees, aligned to ``frequencies``; positive
            values mean the voltage leads the current.
        max_impedance: The maximum of ``|Z(f)|`` over the analysis band in
            kΩ·cm².  For a cell with a genuine resonance this equals
            ``peak_impedance``; for a passive low-pass cell it is the
            lowest-frequency (≈ DC) value, i.e. roughly the input impedance.
            Always populated.
        resonance_frequency: Frequency of the ``|Z|`` resonance peak in Hz, or
            ``None`` when there is no genuine resonance (the maximum lies at a
            band edge, does not clear the prominence margin, or has no
            bracketable −3 dB width).
        peak_impedance: ``|Z|`` at ``resonance_frequency`` in kΩ·cm², or
            ``None`` when there is no genuine resonance.  When non-``None`` it
            is identical to ``max_impedance`` (the resonance peak *is* the
            band maximum).
        quality_factor: Dimensionless ``Q = f_R / FWHM`` where FWHM is the
            full width of the ``|Z|`` peak at the half-power (−3 dB) level, or
            ``None`` when there is no genuine resonance.
        magnitude_mohm: ``|Z(f)|`` converted to absolute MΩ when ``area_cm2``
            is supplied; ``None`` otherwise.
        max_impedance_mohm: ``max_impedance`` in MΩ when ``area_cm2`` is
            supplied; ``None`` otherwise.
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
    max_impedance: float
    resonance_frequency: float | None
    peak_impedance: float | None
    quality_factor: float | None
    magnitude_mohm: list[float] | None = None
    max_impedance_mohm: float | None = None
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
    that bin and its inward neighbor.

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
    FFT bins.  A peak narrower than the FFT frequency resolution is not
    resolved and is rejected (this also screens out lone single-bin spikes,
    e.g. from an interior stimulus dropout).

    Args:
        frequencies: Analysis-band frequency axis in Hz, ascending and
            uniformly spaced.
        magnitude: ``|Z(f)|`` aligned to ``frequencies``.
        peak_idx: Index of the ``|Z|`` peak within the band.

    Returns:
        The quality factor, or ``None`` when either half-power crossing cannot
        be bracketed inside the band, the bracketed width is non-positive, or
        the width is narrower than the frequency resolution.
    """
    half_power = float(magnitude[peak_idx]) / np.sqrt(2.0)
    f_low = _half_power_crossing(frequencies, magnitude, peak_idx, half_power, -1)
    f_high = _half_power_crossing(frequencies, magnitude, peak_idx, half_power, +1)
    if f_low is None or f_high is None:
        return None
    fwhm = f_high - f_low
    df = float(frequencies[1] - frequencies[0])
    if fwhm < df:
        return None
    return float(frequencies[peak_idx]) / fwhm


def _longest_usable_run(values: np.ndarray, threshold: float) -> tuple[int, int]:
    """Return half-open bounds of the longest contiguous ``values >= threshold`` run.

    Args:
        values: 1-D array (here: per-bin stimulus magnitudes).
        threshold: Inclusive lower bound a value must meet to count as usable.

    Returns:
        ``(lo, hi)`` such that ``values[lo:hi]`` is the longest contiguous
        block of indices all meeting ``threshold``.  ``(0, 0)`` when no value
        meets the threshold.
    """
    idx = np.flatnonzero(values >= threshold)
    if idx.size == 0:
        return 0, 0
    # Group boundaries: positions in `idx` where consecutive indices jump by >1.
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [idx.size]))
    k = int(np.argmax(stops - starts))
    return int(idx[starts[k]]), int(idx[stops[k] - 1]) + 1


def _prepare_impedance_window(
    time: np.ndarray,
    voltage: np.ndarray,
    injected_current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    f_start: float,
    f_end: float,
) -> tuple[str, None] | tuple[None, _PreparedWindow]:
    """Validate the chirp inputs and extract the analyzable window.

    Performs the cheap pre-FFT checks (shape / NaN / band sanity, a too-short
    chirp window, and a suprathreshold response that leaves no long-enough
    spike-free segment), and on success returns the windowed traces — the full
    chirp window when the response is subthreshold throughout, or the longest
    contiguous spike-free sub-window otherwise.  The post-FFT failures (no
    stimulus power in the band, too few in-band bins after trimming) are *not*
    detected here.

    Args:
        time: Time axis in ms.
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.

    Returns:
        ``(reason, None)`` with a single-sentence failure reason, or
        ``(None, (t_win, v_win, i_win, analyzed_window_ms))`` on success where
        ``analyzed_window_ms`` is ``None`` for the full chirp window or the
        sub-window duration (ms) when a spike-free segment was used.  Exactly
        one of the two tuple elements is non-``None``.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(injected_current, dtype=float)
    if v.shape != t.shape or i.shape != t.shape:
        return "internal: trace length mismatch.", None
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(i))):
        return "internal: trace contains non-finite values.", None
    if f_end <= f_start or f_start < 0.0:
        return f"invalid frequency band [{f_start}, {f_end}] Hz.", None
    window_ms = stim_end_ms - stim_start_ms
    if window_ms < _MIN_WINDOW_MS:
        return (
            f"the chirp window ({window_ms:.1f} ms) is shorter than the "
            f"minimum {_MIN_WINDOW_MS:.1f} ms needed for a meaningful spectrum."
        ), None
    mask = (t >= stim_start_ms) & (t < stim_end_ms)
    if int(mask.sum()) < 4:
        return "too few samples in the chirp window.", None
    t_win, v_win, i_win = t[mask], v[mask], i[mask]

    # longest_subthreshold_run returns (0, N) when the window has no spikes, so
    # a single call covers both "subthreshold throughout" and "fall back to the
    # widest spike-free segment" — no separate is_subthreshold check needed.
    run = longest_subthreshold_run(t_win, v_win)
    if run is None:
        return (
            "the cell fired throughout the chirp — reduce the amplitude or "
            "apply a hyperpolarizing holding current."
        ), None
    lo, hi = run
    if (lo, hi) == (0, t_win.size):
        return None, (t_win, v_win, i_win, None)
    span = float(t_win[hi - 1] - t_win[lo])
    if span < _MIN_WINDOW_MS:
        return (
            f"the longest spike-free segment ({span:.1f} ms) is shorter than the "
            f"minimum {_MIN_WINDOW_MS:.1f} ms — reduce the amplitude or apply a "
            "hyperpolarizing holding current."
        ), None
    return None, (t_win[lo:hi], v_win[lo:hi], i_win[lo:hi], span)


def impedance_unavailable_reason(
    time: np.ndarray,
    voltage: np.ndarray,
    injected_current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
    f_start: float,
    f_end: float,
) -> str:
    """Return a human-readable reason impedance analysis can't proceed, or ``""``.

    Returns the empty string when the cheap pre-FFT checks pass (so the
    analysis would proceed past the windowing / suprathreshold guards), letting
    callers use the result directly as UI copy.  Only those pre-FFT failure
    modes are reported — when this returns ``""`` and :func:`analyze_impedance`
    still returns ``None``, the caller should fall back to a generic "too little
    usable signal in the band" message.

    Args:
        time: Time axis in ms.
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.

    Returns:
        A single-sentence reason string, or ``""`` when the preamble checks
        pass.
    """
    reason, _ = _prepare_impedance_window(
        time, voltage, injected_current, stim_start_ms, stim_end_ms, f_start, f_end
    )
    return reason or ""


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
    ``Z(f) = V̂(f) / Î(f)`` over the swept band ``[f_start, f_end]`` (keeping
    only the longest contiguous run of FFT bins where the chirp has usable
    spectral power), and reports the magnitude and phase spectra, the
    band-maximum impedance, plus — when a genuine resonance is present — the
    resonance frequency and quality factor.  A genuine resonance requires an
    interior ``|Z|`` maximum that clears the prominence margin over the
    low-frequency reference and has a half-power (−3 dB) width that is both
    bracketable inside the band and at least one FFT bin wide;
    ``resonance_frequency``, ``peak_impedance`` and ``quality_factor`` are
    otherwise all ``None``.

    Membrane impedance is a linear, small-signal quantity, so spike transients
    are excluded: when the windowed response contains spikes the analysis falls
    back to the longest contiguous spike-free sub-window (a linear chirp's
    frequency is time-dependent, so the recovered profile then covers only a
    sub-band of ``[f_start, f_end]`` — surfaced via
    :attr:`ImpedanceProfile.analyzed_window_ms`).  The analysis bails out
    (returns ``None``) when no such spike-free segment is long enough — see
    :func:`impedance_unavailable_reason` for the user-facing reason.

    Args:
        time: Time axis in ms, assumed uniformly sampled (the FFT frequency
            axis is derived from the mean sample interval).
        voltage: Membrane voltage response in mV.
        injected_current: Injected chirp current in µA/cm².
        stim_start_ms: Start of the chirp stimulus window in ms.
        stim_end_ms: End of the chirp stimulus window in ms.
        f_start: Starting frequency of the chirp sweep in Hz.
        f_end: Ending frequency of the chirp sweep in Hz.
        area_cm2: Optional membrane surface area in cm².  When supplied, the
            absolute MΩ counterparts (``magnitude_mohm``, ``max_impedance_mohm``,
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
    # _prepare_impedance_window returns exactly one of (reason, None) /
    # (None, prepared), so a None `prepared` always carries a non-empty reason.
    reason, prepared = _prepare_impedance_window(
        time, voltage, injected_current, stim_start_ms, stim_end_ms, f_start, f_end
    )
    if prepared is None:
        logger.warning("analyze_impedance: %s", reason)
        return None
    t_win, v_win, i_win, analyzed_window_ms = prepared

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
    # Keep only the longest contiguous run of bins with usable stimulus power so
    # that dividing by a near-zero stimulus cannot inflate the band-edge
    # estimate or spike it at an interior dropout.
    lo, hi = _longest_usable_run(i_mag, _MIN_STIM_FRAC * i_max)
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

    max_impedance = float(np.max(mag))
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
        # The resonance peak is the band maximum (peak_idx == argmax(mag)), so
        # peak_impedance == max_impedance here; keep both for callers that want
        # the resonance-specific value to be None when there is no resonance.
        resonance_frequency: float | None = float(fb[peak_idx])
        peak_impedance: float | None = max_impedance
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
    max_impedance_mohm = density_to_absolute_r_in(max_impedance, area_cm2)
    peak_impedance_mohm = (
        density_to_absolute_r_in(peak_impedance, area_cm2)
        if peak_impedance is not None
        else None
    )

    return ImpedanceProfile(
        frequencies=fb.tolist(),
        magnitude=mag.tolist(),
        phase=phase_deg.tolist(),
        max_impedance=max_impedance,
        resonance_frequency=resonance_frequency,
        peak_impedance=peak_impedance,
        quality_factor=quality_factor,
        magnitude_mohm=magnitude_mohm,
        max_impedance_mohm=max_impedance_mohm,
        peak_impedance_mohm=peak_impedance_mohm,
        area_cm2=area_cm2,
        f_start=float(f_start),
        f_end=float(f_end),
        analyzed_window_ms=analyzed_window_ms,
    )
