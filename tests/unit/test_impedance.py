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


def _chirp_and_time(
    duration_ms: float = _DURATION_MS,
    f_start: float = _F_START,
    f_end: float = _F_END,
    amplitude: float = 1.0,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pure_resistor_flat_magnitude() -> None:
    """A pure resistor (V = R·I) yields flat |Z| ≈ R and zero phase."""
    resistance = 50.0  # kΩ·cm²
    time, current = _chirp_and_time()
    voltage = resistance * current

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
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
    time, current = _chirp_and_time(dc_offset=5.0)
    voltage = resistance * current + 12.0

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
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
    voltage = _apply_transfer_function(current, transfer)

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
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
    voltage = _apply_transfer_function(current, transfer)

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
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


def test_band_restriction() -> None:
    """The analysis band is clipped to [f_start, f_end]."""
    time, current = _chirp_and_time()
    voltage = 10.0 * current

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, 10.0, 50.0
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
    voltage = 10.0 * current

    result = analyze_impedance(time, voltage, current, 0.0, 10.0, _F_START, _F_END)

    assert result is None


def test_returns_none_invalid_band() -> None:
    """A band with f_end <= f_start returns None."""
    time, current = _chirp_and_time()
    voltage = 10.0 * current

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, 50.0, 50.0
    )

    assert result is None


def test_returns_none_with_nans() -> None:
    """Non-finite values in the voltage trace return None."""
    time, current = _chirp_and_time()
    voltage = 10.0 * current
    voltage[100] = np.nan

    result = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
    )

    assert result is None


def test_returns_none_too_few_band_bins() -> None:
    """A band wide enough in Hz but with too few FFT bins returns None."""
    time, current = _chirp_and_time(duration_ms=200.0)
    voltage = 10.0 * current

    # 200 ms at 1 kHz → 5 Hz frequency resolution → a 1 Hz band holds < 8 bins.
    result = analyze_impedance(time, voltage, current, 0.0, 200.0, 49.0, 50.0)

    assert result is None


def test_absolute_conversion_with_area() -> None:
    """Supplying area_cm2 fills the absolute MΩ spectrum; omitting it leaves it None."""
    resistance = 40.0  # kΩ·cm²
    area_cm2 = 2.0e-5
    time, current = _chirp_and_time()
    voltage = resistance * current

    with_area = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END, area_cm2
    )
    without_area = analyze_impedance(
        time, voltage, current, 0.0, time[-1] + _DT_MS, _F_START, _F_END
    )

    assert with_area is not None and without_area is not None
    assert without_area.magnitude_mohm is None
    assert with_area.magnitude_mohm is not None
    assert len(with_area.magnitude_mohm) == len(with_area.magnitude)
    expected = resistance / area_cm2 / 1000.0
    assert np.asarray(with_area.magnitude_mohm) == pytest.approx(expected, rel=1e-6)
    assert with_area.area_cm2 == area_cm2
