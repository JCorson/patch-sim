"""Tests for CSV export of simulation results (patch_sim_ui/export.py)."""

import io
from datetime import datetime

import numpy as np
import pytest

import patch_sim
from patch_sim.constants import CURRENT_CLAMP, VOLTAGE_CLAMP
from patch_sim_ui.export import (
    _slugify,
    build_export_metadata,
    export_filename,
    sweeps_to_csv,
)
from patch_sim_ui.sweep import Sweep


def _sweep(
    n: int = 4,
    *,
    mode: str = CURRENT_CLAMP,
    label: str = "",
    offset: float = 0.0,
    currents: dict[str, list[float]] | None = None,
    gating: dict[str, list[float]] | None = None,
) -> Sweep:
    """Return a hand-built sweep of *n* samples.

    Args:
        n: Number of samples.
        mode: Clamp mode.
        label: Sweep label.
        offset: Value added to every trace so sweeps are distinguishable.
        currents: Channel currents; defaults to a single ``INa`` trace.
        gating: Gating traces; defaults to ``m`` and ``ca_i``.

    Returns:
        A populated sweep.
    """
    base = [offset + i for i in range(n)]
    return Sweep(
        label=label,
        color="",
        time=[0.025 * i for i in range(n)],
        voltage=[v - 65.0 for v in base],
        dvdt=base,
        total_current=[v * 0.5 for v in base],
        stimulus=[v + 10.0 for v in base],
        clamp_mode=mode,
        channel_currents=currents if currents is not None else {"INa": base},
        channel_gating=gating
        if gating is not None
        else {"m": [0.1] * n, "ca_i": [1e-4] * n},
    )


def _header(csv: str) -> list[str]:
    """Return the column names of *csv*.

    Args:
        csv: CSV text with ``#`` metadata lines.

    Returns:
        Column names from the first non-comment line.
    """
    line = next(ln for ln in csv.splitlines() if not ln.startswith("#"))
    return line.split(",")


def test_current_clamp_single_sweep_columns() -> None:
    """A single CC sweep has no suffix and exports V, Istim, currents, gating, ca_i."""
    csv = sweeps_to_csv([_sweep()])
    assert _header(csv) == [
        "time_ms",
        "V_mV",
        "Istim_uA_cm2",
        "Itotal_uA_cm2",
        "INa_uA_cm2",
        "m",
        "ca_i_mM",
    ]


def test_voltage_clamp_exports_command_once() -> None:
    """VC exports Vcmd_mV once and no separate voltage or dvdt column."""
    csv = sweeps_to_csv([_sweep(mode=VOLTAGE_CLAMP)])
    header = _header(csv)
    assert header[:3] == ["time_ms", "Vcmd_mV", "Itotal_uA_cm2"]
    assert "V_mV" not in header
    assert not any("dvdt" in h for h in header)


def test_multi_sweep_columns_are_quantity_major_with_suffix() -> None:
    """Multiple sweeps get zero-padded suffixes, grouped by quantity."""
    csv = sweeps_to_csv([_sweep(), _sweep(offset=5.0)])
    header = _header(csv)
    assert header[:5] == [
        "time_ms",
        "V_mV_s01",
        "V_mV_s02",
        "Istim_uA_cm2_s01",
        "Istim_uA_cm2_s02",
    ]


def test_suffix_widens_beyond_ninety_nine_sweeps() -> None:
    """Suffix padding grows so column names stay sortable for 100+ sweeps."""
    csv = sweeps_to_csv([_sweep(n=1, offset=float(i)) for i in range(100)])
    header = _header(csv)
    assert "V_mV_s001" in header
    assert "V_mV_s100" in header


def test_round_trip_values() -> None:
    """Data read back via genfromtxt match the sweep values."""
    sweeps = [_sweep(offset=0.0), _sweep(offset=5.0)]
    csv = sweeps_to_csv(sweeps)
    body = "\n".join(ln for ln in csv.splitlines() if not ln.startswith("#"))
    arr = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
    assert arr["time_ms"] == pytest.approx(sweeps[0].time)
    assert arr["V_mV_s01"] == pytest.approx(sweeps[0].voltage)
    assert arr["V_mV_s02"] == pytest.approx(sweeps[1].voltage)
    assert arr["ca_i_mM_s01"] == pytest.approx(1e-4)


def test_uneven_lengths_leave_empty_cells() -> None:
    """A shorter sweep yields empty trailing cells; time comes from the longest."""
    csv = sweeps_to_csv([_sweep(n=2), _sweep(n=4)])
    rows = [ln for ln in csv.splitlines() if not ln.startswith("#")]
    assert len(rows) == 1 + 4
    header = rows[0].split(",")
    last = rows[-1].split(",")
    assert last[0] == "0.0750"
    assert last[header.index("V_mV_s01")] == ""
    assert last[header.index("V_mV_s02")] != ""


def test_channel_union_leaves_missing_channel_empty() -> None:
    """A channel present in only one sweep is exported, empty for the other."""
    a = _sweep(currents={"INa": [1.0] * 4})
    b = _sweep(currents={"IK": [2.0] * 4})
    csv = sweeps_to_csv([a, b])
    header = _header(csv)
    assert header.index("INa_uA_cm2_s01") < header.index("IK_uA_cm2_s01")
    first = [ln for ln in csv.splitlines() if not ln.startswith("#")][1].split(",")
    assert first[header.index("INa_uA_cm2_s02")] == ""
    assert float(first[header.index("IK_uA_cm2_s02")]) == pytest.approx(2.0)


def test_empty_sweeps_raise() -> None:
    """Exporting no sweeps is an error."""
    with pytest.raises(ValueError):
        sweeps_to_csv([])


def test_metadata_block_is_ascii_and_precedes_header() -> None:
    """Metadata lines are ASCII, transliterate units, and precede the header."""
    meta = [("sweep s01", "+10.0 µA/cm²"), ("tau", "τ 5 MΩ")]
    csv = sweeps_to_csv([_sweep()], meta)
    csv.encode("ascii")
    lines = csv.splitlines()
    assert lines[0] == "# patch_sim export"
    assert "# sweep s01: +10.0 uA/cm2" in lines
    assert "# tau: tau 5 MOhm" in lines
    assert not lines[3].startswith("#")


def test_metadata_value_cannot_span_lines() -> None:
    """Newlines in metadata values do not break out of the comment line."""
    csv = sweeps_to_csv([_sweep()], [("label", "a\nb")])
    assert "# label: a b" in csv.splitlines()


def test_build_export_metadata_contents() -> None:
    """Metadata records provenance, scalars, per-channel g_max and sweep labels."""
    neuron = patch_sim.Neuron()
    meta = dict(
        build_export_metadata(
            neuron,
            neuron_preset="Squid",
            protocol_preset="I-V",
            clamp_mode=VOLTAGE_CLAMP,
            protocol_type="Step",
            protocol_params={"stimulus_duration_ms": 30.0},
            sweep_labels=["-10 mV", "+0 mV"],
            continuous=True,
            simulated_at=datetime(2026, 1, 2, 3, 4, 5).astimezone(),
        )
    )
    assert meta["clamp_mode"] == VOLTAGE_CLAMP
    assert meta["neuron_preset"] == "Squid"
    assert meta["dt_ms"] == "0.025"
    assert meta["sampling_hz"] == "40000"
    assert meta["stimulus_duration_ms"] == "30"
    assert meta["simulated_at"].startswith("2026-01-02T03:04:05")
    assert meta["continuous_mode"] == "true"
    assert "exported_at" not in meta
    assert meta["sweep s01"] == "-10 mV"
    assert meta["sweep s02"] == "+0 mV"
    assert "C_m" in meta


def test_nan_in_data_is_written_as_empty_cell() -> None:
    """A NaN produced by the simulation becomes an empty cell, not 'nan'."""
    sweep = _sweep()
    sweep.voltage[1] = float("nan")
    csv = sweeps_to_csv([sweep])
    assert "nan" not in csv.lower()
    rows = [ln for ln in csv.splitlines() if not ln.startswith("#")]
    header = rows[0].split(",")
    cell = rows[2].split(",")[header.index("V_mV")]
    assert cell == ""


def test_export_filename_format() -> None:
    """Filenames embed slugged names and the timestamp."""
    name = export_filename(
        "Squid Giant Axon (Classic HH)",
        "Action Potential",
        "csv",
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert name == (
        "patch-sim_squid-giant-axon-classic-hh_action-potential_20260102-030405.csv"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Squid Giant Axon (Classic HH)", "squid-giant-axon-classic-hh"),
        ("Café Neuron", "cafe-neuron"),
        ("", "unnamed"),
        ("???", "unnamed"),
        ('a"b/c\\d\r\ne', "a-b-c-d-e"),
        ("x" * 100, "x" * 40),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    """Slugs are lowercase ASCII, hyphen separated, bounded and never empty.

    Args:
        text: Input text.
        expected: Expected slug.
    """
    assert _slugify(text) == expected
