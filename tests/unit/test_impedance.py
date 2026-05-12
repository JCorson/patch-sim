"""Unit tests for patch_sim.analysis.impedance.

Covers analyze_impedance with synthetic stimulus/response signals and edge
cases.  Integration tests against real simulations live in
tests/integration/test_impedance_simulation.py.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.analysis.impedance import analyze_impedance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS_HZ = 1000.0  # Sampling rate for the synthetic signals.
_DT_MS = 1000.0 / _FS_HZ
_DURATION_MS = 4000.0
_F_START = 1.0
_F_END = 100.0
_AMP = 0.05  # Small chirp amplitude so |Z|·amplitude stays subthreshold.
_BASELINE_MV = -65.0  # Physiological resting potential for the response trace.


def _chirp_and_time(
    duration_ms: float = _DURATION_MS,
    f_start: float = _F_START,
    f_end: float = _F_END,
    amplitude: float = _AMP,
    dc_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a chirp stimulus and matching time axis at ``_FS_HZ``.

    Args:
        duration_ms: Total duration in ms.
        f_start: Starting frequency of the sweep in Hz.
        f_end: Ending frequency of the sweep in Hz.
        amplitude: Chirp amplitude.
        dc_offset: DC offset added to the chirp.

    Returns:
        Tuple of (time array in ms, chirp current array).
    """
    current = patch_sim.chirp_current(
        duration=duration_ms,
        dc_offset=dc_offset,
        amplitude=amplitude,
        start_frequency=f_start,
        end_frequency=f_end,
        sampling_frequency=_FS_HZ,
    )
    time = np.arange(current.size, dtype=float) * _DT_MS
    return time, current


def _apply_transfer_function(current: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Apply a frequency-domain transfer function to a real signal.

    Args:
        current: Real-valued input signal.
        transfer: Complex transfer function evaluated at ``rfftfreq`` bins.

    Returns:
        The filtered real-valued output signal, same length as ``current``.
    """
    return np.fft.irfft(transfer * np.fft.rfft(current), n=current.size)


def _window_end(time: np.ndarray) -> float:
    """Return a window-end time that includes the final sample of ``time``.

    Args:
        time: Ascending time axis in ms.

    Returns:
        A value slightly past ``time[-1]`` so the ``t < stim_end`` mask keeps
        every sample.
    """
    return float(time[-1]) + _DT_MS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pure_resistor_flat_magnitude() -> None:
    """A pure resistor (V = R·I + V_rest) yields flat |Z| ≈ R and zero phase."""
    resistance = 50.0  # kΩ·cm²
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + resistance * current

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    assert result.resonance_frequency is None
    assert result.peak_impedance is None
    assert result.quality_factor is None
    mag = np.asarray(result.magnitude)
    phase = np.asarray(result.phase)
    assert mag == pytest.approx(resistance, rel=1e-6)
    assert phase == pytest.approx(0.0, abs=1e-6)


def test_resistor_with_dc_offset_subtracted() -> None:
    """DC offsets on V and I are removed before the FFT, leaving |Z| ≈ R."""
    resistance = 30.0  # kΩ·cm²
    time, chirp_ac = _chirp_and_time()
    current = chirp_ac + 5.0  # I carries a DC offset of 5.0.
    voltage = _BASELINE_MV + resistance * chirp_ac  # V carries a different offset.

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    assert np.asarray(result.magnitude) == pytest.approx(resistance, rel=1e-6)
    assert np.asarray(result.phase) == pytest.approx(0.0, abs=1e-6)


def test_first_order_lowpass_no_interior_peak() -> None:
    """A one-pole low-pass response has its |Z| max at the band edge, no f_R."""
    tau_s = 0.05  # 50 ms membrane time constant
    time, current = _chirp_and_time()
    freqs = np.fft.rfftfreq(current.size, d=1.0 / _FS_HZ)
    transfer = 1.0 / (1.0 + 1j * 2.0 * np.pi * freqs * tau_s)
    voltage = _BASELINE_MV + _apply_transfer_function(current, transfer)

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    assert result.resonance_frequency is None
    assert result.quality_factor is None
    mag = np.asarray(result.magnitude)
    assert int(np.argmax(mag)) == 0
    assert mag[0] > mag[-1]


def test_synthetic_resonance_detects_f_r() -> None:
    """A second-order band-pass response yields an interior f_R and finite Q."""
    f0 = 8.0
    q0 = 2.0
    time, current = _chirp_and_time()
    freqs = np.fft.rfftfreq(current.size, d=1.0 / _FS_HZ)
    ratio = freqs / f0
    transfer = 1.0 / (1.0 - ratio**2 + 1j * ratio / q0)
    voltage = _BASELINE_MV + _apply_transfer_function(current, transfer)

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    assert result.resonance_frequency == pytest.approx(f0, abs=2.0)
    assert result.peak_impedance is not None
    assert result.quality_factor is not None
    assert result.quality_factor > 0.3
    # The interior peak should clearly exceed the band edges.
    mag = np.asarray(result.magnitude)
    assert result.peak_impedance > mag[0]
    assert result.peak_impedance > mag[-1]


def test_shallow_interior_bump_reports_no_resonance() -> None:
    """A barely-underdamped 2nd-order low-pass has an interior |Z| max but no resonance.

    With ``Q0`` just above the ``1/√2`` threshold the response has a genuine
    interior maximum, but it rises only a fraction of a percent above the
    low-frequency reference — well below the prominence margin — so
    ``resonance_frequency``, ``peak_impedance`` and ``quality_factor`` are all
    ``None`` together (a "resonance" with no measurable −3 dB width is just
    ripple on a passive response).  The magnitude/phase spectra are still
    populated.
    """
    f0 = 30.0
    q0 = 0.72  # just above 1/sqrt(2) ≈ 0.707 → a tiny interior peak exists
    time, current = _chirp_and_time()
    freqs = np.fft.rfftfreq(current.size, d=1.0 / _FS_HZ)
    ratio = freqs / f0
    transfer = 1.0 / (1.0 - ratio**2 + 1j * ratio / q0)
    voltage = _BASELINE_MV + _apply_transfer_function(current, transfer)

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    mag = np.asarray(result.magnitude)
    # There IS an interior maximum...
    assert 0 < int(np.argmax(mag)) < mag.size - 1
    # ...but it is too shallow to be reported as a resonance.
    assert result.resonance_frequency is None
    assert result.peak_impedance is None
    assert result.quality_factor is None
    assert len(result.magnitude) == len(result.phase) > 0


def test_band_restriction() -> None:
    """The analysis band is clipped to [f_start, f_end]."""
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + 10.0 * current

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), 10.0, 50.0
    )

    assert result is not None
    assert result.f_start == 10.0
    assert result.f_end == 50.0
    freqs = np.asarray(result.frequencies)
    assert freqs.min() >= 10.0
    assert freqs.max() <= 50.0


def test_returns_none_short_window() -> None:
    """A chirp window shorter than the minimum returns None."""
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + 10.0 * current

    result = analyze_impedance(time, voltage, current, 0.0, 10.0, _F_START, _F_END)

    assert result is None


def test_returns_none_invalid_band() -> None:
    """A band with f_end <= f_start returns None."""
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + 10.0 * current

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), 50.0, 50.0
    )

    assert result is None


def test_returns_none_with_nans() -> None:
    """Non-finite values in the voltage trace return None."""
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + 10.0 * current
    voltage[100] = np.nan

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is None


def test_returns_none_too_few_band_bins() -> None:
    """A band wide enough in Hz but with too few FFT bins returns None."""
    time, current = _chirp_and_time(duration_ms=200.0)
    voltage = _BASELINE_MV + 10.0 * current

    # 200 ms at 1 kHz → 5 Hz frequency resolution → a 1 Hz band holds < 8 bins.
    result = analyze_impedance(time, voltage, current, 0.0, 200.0, 49.0, 50.0)

    assert result is None


def test_returns_none_when_spikes_throughout() -> None:
    """Spikes covering the entire window leave no spike-free segment → None.

    A single isolated spike is now tolerated via the spike-free sub-window
    fallback (covered by ``test_recovers_profile_from_spike_free_segment``); to
    exercise the genuine "no usable segment" path we tile spikes every 50 ms
    so every excised guard window butts against the next.
    """
    time, current = _chirp_and_time(amplitude=1.0)
    voltage = np.full_like(time, _BASELINE_MV)
    # Spike every 50 ms (= _REFRACTORY_SAMPLES at this 1 kHz sample rate, so
    # each event is detected) across the entire trace.
    for centre_ms in np.arange(50.0, _DURATION_MS, 50.0):
        spike_window = (time >= centre_ms) & (time < centre_ms + 5.0)
        voltage[spike_window] = 30.0

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is None


def test_recovers_profile_from_spike_free_segment() -> None:
    """A brief spike + a quiet resonator recovers via the sub-window fallback.

    The full chirp window contains a spike, but the spike-free remainder is long
    enough (≫ the 50 ms minimum) to carry an FFT, so ``analyze_impedance``
    returns a profile and reports the analyzed sub-window duration.  The
    reported band is narrower than the requested band because the recovered
    segment covers only the late portion of the linear frequency sweep.
    """
    f0 = 8.0
    q0 = 2.0
    time, current = _chirp_and_time()
    freqs = np.fft.rfftfreq(current.size, d=1.0 / _FS_HZ)
    ratio = freqs / f0
    transfer = 1.0 / (1.0 - ratio**2 + 1j * ratio / q0)
    voltage = _BASELINE_MV + _apply_transfer_function(current, transfer)
    # Inject one synthetic spike near the start of the trace.
    spike_window = (time >= 100.0) & (time < 105.0)
    voltage[spike_window] = 30.0

    result = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert result is not None
    assert result.analyzed_window_ms is not None
    # The recovered window covers everything past the spike + guard padding,
    # which on a 4000 ms trace is well over 3500 ms.
    assert result.analyzed_window_ms > _DURATION_MS - 200.0
    # ``f_start`` / ``f_end`` report the requested band; the actually-covered
    # band lives in ``frequencies``.
    assert result.f_start == _F_START
    assert result.f_end == _F_END
    assert min(result.frequencies) >= _F_START


def test_impedance_unavailable_reason_messages() -> None:
    """The public reason wrapper returns "" for OK inputs and a sentence otherwise."""
    time, current = _chirp_and_time()
    clean_voltage = _BASELINE_MV + 10.0 * current

    # OK case → empty string.
    assert (
        patch_sim.impedance_unavailable_reason(
            time, clean_voltage, current, 0.0, _window_end(time), _F_START, _F_END
        )
        == ""
    )

    # Window shorter than _MIN_WINDOW_MS → "shorter than the minimum".
    short_reason = patch_sim.impedance_unavailable_reason(
        time, clean_voltage, current, 0.0, 10.0, _F_START, _F_END
    )
    assert "shorter than the minimum" in short_reason

    # Spikes throughout → "fired throughout" message.
    busy_voltage = np.full_like(time, _BASELINE_MV)
    for centre_ms in np.arange(50.0, _DURATION_MS, 50.0):
        sp = (time >= centre_ms) & (time < centre_ms + 5.0)
        busy_voltage[sp] = 30.0
    busy_reason = patch_sim.impedance_unavailable_reason(
        time, busy_voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )
    assert "fired throughout" in busy_reason or "spike-free segment" in busy_reason


def test_absolute_conversion_with_area() -> None:
    """Supplying area_cm2 fills the absolute MΩ spectrum; omitting it leaves it None."""
    resistance = 40.0  # kΩ·cm²
    area_cm2 = 2.0e-5
    time, current = _chirp_and_time()
    voltage = _BASELINE_MV + resistance * current

    with_area = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END, area_cm2
    )
    without_area = analyze_impedance(
        time, voltage, current, 0.0, _window_end(time), _F_START, _F_END
    )

    assert with_area is not None and without_area is not None
    assert without_area.magnitude_mohm is None
    assert with_area.magnitude_mohm is not None
    assert len(with_area.magnitude_mohm) == len(with_area.magnitude)
    expected = resistance / area_cm2 / 1000.0
    assert np.asarray(with_area.magnitude_mohm) == pytest.approx(expected, rel=1e-6)
    assert with_area.area_cm2 == area_cm2
