"""CSV export of simulation results (no Reflex dependency).

The CSV is wide: one row per time sample, one column per
(quantity, sweep) pair.  Columns are ordered quantity-major and
sweep-minor so that each quantity's sweeps sit side by side::

    time_ms, V_mV_s01, V_mV_s02, Istim_uA_cm2_s01, Istim_uA_cm2_s02, ...

A single sweep carries no ``_sNN`` suffix.  Voltage-clamp sweeps export the
command voltage once (``Vcmd_mV``) because the recorded voltage equals the
command.  Every computed trace is exported regardless of plot visibility.

A ``# key: value`` comment block precedes the header row and records the
neuron, protocol, and sampling parameters that produced the data.  All
headers and metadata are ASCII so the file opens cleanly in any tool.
"""

import io
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from importlib import metadata as importlib_metadata

import numpy as np

import patch_sim
from patch_sim.constants import SIM_SAMPLING_FREQ, VOLTAGE_CLAMP
from patch_sim_ui import presets
from patch_sim_ui.sweep import Sweep

#: Unicode characters with a conventional ASCII spelling in unit strings.
_ASCII_SUBSTITUTIONS: dict[str, str] = {
    "µ": "u",
    "μ": "u",
    "²": "2",
    "Ω": "Ohm",
    "τ": "tau",
}

#: Gating-dict key holding intracellular calcium concentration (mM).
_CA_I_KEY = "ca_i"


def _ascii(text: str) -> str:
    """Return *text* reduced to single-line ASCII.

    Known unit symbols are transliterated (``µ`` to ``u``, ``²`` to ``2``,
    ``Ω`` to ``Ohm``, ``τ`` to ``tau``); anything else non-ASCII becomes
    ``?``.  Line breaks become spaces so a value cannot span lines.

    Args:
        text: Arbitrary text.

    Returns:
        ASCII-only, single-line text.
    """
    for src, dst in _ASCII_SUBSTITUTIONS.items():
        text = text.replace(src, dst)
    text = " ".join(text.splitlines())
    return text.encode("ascii", "replace").decode("ascii")


def _slugify(text: str, max_len: int = 40) -> str:
    """Return a filename-safe slug of *text*.

    Args:
        text: Arbitrary text (e.g. a neuron preset name).
        max_len: Maximum slug length.

    Returns:
        Lowercase ``[a-z0-9-]`` slug, or ``"unnamed"`` when nothing remains.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    slug = slug[:max_len].strip("-")
    return slug or "unnamed"


def export_filename(neuron_preset: str, protocol: str, ext: str, now: datetime) -> str:
    """Return the download filename for an export.

    Args:
        neuron_preset: Neuron preset name.
        protocol: Protocol preset name (or protocol type when no preset).
        ext: File extension without the dot (e.g. ``"csv"``).
        now: Timestamp to embed.

    Returns:
        ``patch-sim_<neuron>_<protocol>_<YYYYmmdd-HHMMSS>.<ext>``.
    """
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return f"patch-sim_{_slugify(neuron_preset)}_{_slugify(protocol)}_{stamp}.{ext}"


def _sweep_suffix(index: int, n_sweeps: int) -> str:
    """Return the column suffix identifying sweep *index*.

    Args:
        index: Zero-based sweep index.
        n_sweeps: Total number of sweeps.

    Returns:
        Empty string for a single sweep, otherwise ``_sNN`` zero-padded to
        at least two digits.
    """
    if n_sweeps <= 1:
        return ""
    width = max(2, len(str(n_sweeps)))
    return f"_s{index + 1:0{width}d}"


def _column_specs(
    sweeps: Sequence[Sweep],
) -> list[tuple[str, Callable[[Sweep], Sequence[float] | None]]]:
    """Return the per-quantity column layout shared by all sweeps.

    Channel keys form the union across sweeps in first-seen order.  Each
    getter returns the sweep's values for that quantity, or ``None`` when the
    sweep lacks it (the column is then exported empty).

    Args:
        sweeps: Non-empty sequence of sweeps.

    Returns:
        List of ``(header_base, getter)`` pairs in export order.
    """
    voltage_clamp = sweeps[0].clamp_mode == VOLTAGE_CLAMP
    specs: list[tuple[str, Callable[[Sweep], Sequence[float] | None]]] = []
    if voltage_clamp:
        specs.append(("Vcmd_mV", lambda s: s.stimulus))
    else:
        specs.append(("V_mV", lambda s: s.voltage))
        specs.append(("Istim_uA_cm2", lambda s: s.stimulus))
    specs.append(("Itotal_uA_cm2", lambda s: s.total_current))

    current_keys: list[str] = []
    gating_keys: list[str] = []
    for sweep in sweeps:
        for key in sweep.channel_currents:
            if key not in current_keys:
                current_keys.append(key)
        for key in sweep.channel_gating:
            if key != _CA_I_KEY and key not in gating_keys:
                gating_keys.append(key)
    has_ca = any(_CA_I_KEY in s.channel_gating for s in sweeps)

    for key in current_keys:
        specs.append((f"{key}_uA_cm2", lambda s, k=key: s.channel_currents.get(k)))
    for key in gating_keys:
        specs.append((key, lambda s, k=key: s.channel_gating.get(k)))
    if has_ca:
        specs.append(("ca_i_mM", lambda s: s.channel_gating.get(_CA_I_KEY)))
    return specs


def build_export_metadata(
    neuron: "patch_sim.Neuron",
    *,
    neuron_preset: str,
    protocol_preset: str,
    clamp_mode: str,
    protocol_type: str,
    protocol_params: Mapping[str, float],
    sweep_labels: Sequence[str],
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    """Return ``(key, value)`` metadata describing how the data were produced.

    Args:
        neuron: The neuron that was simulated.
        neuron_preset: Neuron preset name.
        protocol_preset: Protocol preset name (may be empty).
        clamp_mode: Clamp mode that was run.
        protocol_type: Protocol type that was run.
        protocol_params: Protocol timing and stimulus parameters keyed by
            the name to record (e.g. ``"stimulus_duration_ms"``).
        sweep_labels: One label per exported sweep, in export order.
        now: Export timestamp; defaults to the current local time.

    Returns:
        Ordered ``(key, value)`` pairs.  Keys and values are not yet
        ASCII-sanitized; :func:`sweeps_to_csv` does that when writing.
    """
    now = now or datetime.now().astimezone()
    try:
        version = importlib_metadata.version("patch_sim")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"

    meta: list[tuple[str, str]] = [
        ("patch_sim_version", version),
        ("exported_at", now.isoformat(timespec="seconds")),
        ("clamp_mode", clamp_mode),
        ("neuron_preset", neuron_preset),
        ("protocol_preset", protocol_preset),
        ("protocol_type", protocol_type),
        ("dt_ms", f"{1000.0 / SIM_SAMPLING_FREQ:g}"),
        ("sampling_hz", f"{SIM_SAMPLING_FREQ:g}"),
    ]
    meta.extend((key, f"{value:g}") for key, value in protocol_params.items())
    meta.extend(
        (name, f"{getattr(neuron, name):g}")
        for name in presets.NEURON_CONFIG_SCALAR_FIELDS
    )
    meta.extend((f"g_max {ch.name}", f"{ch.g_max:g} mS/cm2") for ch in neuron.channels)
    if len(sweep_labels) > 1:
        n = len(sweep_labels)
        meta.extend(
            (f"sweep {_sweep_suffix(i, n)[1:]}", label)
            for i, label in enumerate(sweep_labels)
        )
    return meta


def sweeps_to_csv(
    sweeps: Sequence[Sweep],
    metadata: Sequence[tuple[str, str]] = (),
) -> str:
    """Serialize *sweeps* to a wide CSV string.

    Time is taken from the longest sweep; shorter sweeps leave their trailing
    cells empty.  Time uses four decimals and data six significant digits.

    Args:
        sweeps: Sweeps to export, all in one clamp mode.
        metadata: ``(key, value)`` pairs written as ``# key: value`` lines
            ahead of the header row.

    Returns:
        The CSV text, newline-terminated, with LF line endings.

    Raises:
        ValueError: If *sweeps* is empty.
    """
    if not sweeps:
        raise ValueError("No sweeps to export.")

    n_sweeps = len(sweeps)
    n_rows = max(len(s.time) for s in sweeps)
    longest = max(sweeps, key=lambda s: len(s.time))
    specs = _column_specs(sweeps)

    headers = ["time_ms"]
    columns = [np.asarray(longest.time, dtype=float)]
    for base, getter in specs:
        for i, sweep in enumerate(sweeps):
            headers.append(base + _sweep_suffix(i, n_sweeps))
            col = np.full(n_rows, np.nan)
            values = getter(sweep)
            if values is not None:
                arr = np.asarray(values, dtype=float)[:n_rows]
                col[: len(arr)] = arr
            columns.append(col)

    buf = io.StringIO()
    buf.write("# patch_sim export\n")
    for key, value in metadata:
        buf.write(f"# {_ascii(key)}: {_ascii(value)}\n")
    buf.write(",".join(headers) + "\n")
    body = io.StringIO()
    np.savetxt(
        body,
        np.column_stack(columns),
        fmt=["%.4f"] + ["%.6g"] * (len(columns) - 1),
        delimiter=",",
        newline="\n",
    )
    buf.write(re.sub(r"(?<=,)nan(?=,|$)", "", body.getvalue(), flags=re.MULTILINE))
    return buf.getvalue()
