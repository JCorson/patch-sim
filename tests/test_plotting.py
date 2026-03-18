"""Unit tests for ap_sim_ui/plotting.py.

Covers Sweep.from_dataframe, build_figure, and _build_hover_tables.
All three are pure functions with no Reflex dependency.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ap_sim_ui.plotting import Sweep, _build_hover_tables, build_figure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 50  # default number of time points for test DataFrames


def _make_df(
    extra_cols: dict[str, list[float]] | None = None,
    n: int = _N,
) -> pd.DataFrame:
    """Return a minimal simulation DataFrame with classic columns.

    Args:
        extra_cols: Additional columns to include beyond the classic set.
        n: Number of time points.

    Returns:
        A DataFrame indexed by time (ms) with classic simulation columns.
    """
    t = np.linspace(0.0, 50.0, n)
    data: dict[str, list[float]] = {
        "voltage": list(np.full(n, -65.0)),
        "total_current": list(np.zeros(n)),
        "sodium_current": list(np.zeros(n)),
        "potassium_current": list(np.zeros(n)),
        "leak_current": list(np.zeros(n)),
        "potassium_activation": list(np.full(n, 0.3)),
        "sodium_activation": list(np.full(n, 0.05)),
        "sodium_inactivation": list(np.full(n, 0.6)),
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data, index=t)


def _make_stimulus(n: int = _N) -> np.ndarray:
    """Return a zero stimulus array of length n.

    Args:
        n: Number of time points.

    Returns:
        A NumPy array of zeros.
    """
    return np.zeros(n)


def _make_sweep(
    label: str = "sweep",
    color: str = "#888888",
    mode: str = "Current Clamp",
    n: int = _N,
    extra_cols: dict[str, list[float]] | None = None,
) -> Sweep:
    """Construct a Sweep via from_dataframe for use in figure tests.

    Args:
        label: Sweep label string.
        color: Hex colour string.
        mode: Clamp mode string.
        n: Number of time points.
        extra_cols: Extra DataFrame columns to include.

    Returns:
        A fully-populated Sweep instance.
    """
    df = _make_df(extra_cols=extra_cols, n=n)
    stim = _make_stimulus(n)
    return Sweep.from_dataframe(df, stim, label, color, mode)


# ---------------------------------------------------------------------------
# Sweep.from_dataframe — classic column classification
# ---------------------------------------------------------------------------


def test_from_dataframe_classic_columns_are_populated() -> None:
    """Classic columns are stored correctly in the Sweep fields."""
    df = _make_df()
    stim = _make_stimulus()
    s = Sweep.from_dataframe(df, stim, "A", "#fff", "Current Clamp")
    assert len(s.voltage) == _N
    assert len(s.sodium_current) == _N
    assert len(s.potassium_current) == _N
    assert len(s.leak_current) == _N
    assert len(s.total_current) == _N
    assert len(s.potassium_activation) == _N
    assert len(s.sodium_activation) == _N
    assert len(s.sodium_inactivation) == _N


def test_from_dataframe_time_index_stored() -> None:
    """The DataFrame index is stored as the time axis."""
    df = _make_df()
    stim = _make_stimulus()
    s = Sweep.from_dataframe(df, stim, "", "", "Current Clamp")
    assert s.time == pytest.approx(df.index.tolist())


def test_from_dataframe_stimulus_stored() -> None:
    """The stimulus array is stored as sweep.stimulus."""
    stim = np.linspace(0, 10, _N)
    df = _make_df()
    s = Sweep.from_dataframe(df, stim, "", "", "Current Clamp")
    assert s.stimulus == pytest.approx(stim.tolist())


def test_from_dataframe_no_extra_columns_gives_empty_dicts() -> None:
    """With only classic columns both additional dicts are empty."""
    s = Sweep.from_dataframe(_make_df(), _make_stimulus(), "", "", "Current Clamp")
    assert s.additional_currents == {}
    assert s.additional_gating == {}


def test_from_dataframe_current_suffix_goes_to_additional_currents() -> None:
    """Extra columns ending with _current are placed in additional_currents."""
    extra = {"ih_current": list(np.ones(_N) * 0.5)}
    s = Sweep.from_dataframe(_make_df(extra_cols=extra), _make_stimulus(), "", "", "CC")
    assert "ih" in s.additional_currents
    assert "ih_current" not in s.additional_currents
    assert s.additional_currents["ih"] == pytest.approx([0.5] * _N)


def test_from_dataframe_current_suffix_stripped_correctly() -> None:
    """The _current suffix is fully stripped to produce the channel key."""
    extra = {"foo_current": list(np.zeros(_N))}
    s = Sweep.from_dataframe(_make_df(extra_cols=extra), _make_stimulus(), "", "", "CC")
    assert list(s.additional_currents.keys()) == ["foo"]


def test_from_dataframe_non_current_extra_goes_to_additional_gating() -> None:
    """Extra columns without _current suffix are placed in additional_gating."""
    extra = {"r": list(np.full(_N, 0.4))}
    s = Sweep.from_dataframe(_make_df(extra_cols=extra), _make_stimulus(), "", "", "CC")
    assert "r" in s.additional_gating
    assert "r" not in s.additional_currents
    assert s.additional_gating["r"] == pytest.approx([0.4] * _N)


def test_from_dataframe_multiple_extra_columns_classified() -> None:
    """Multiple extra columns are each classified into the correct dict."""
    extra = {
        "ika_current": list(np.ones(_N)),
        "a": list(np.full(_N, 0.1)),
        "b": list(np.full(_N, 0.9)),
    }
    s = Sweep.from_dataframe(_make_df(extra_cols=extra), _make_stimulus(), "", "", "CC")
    assert set(s.additional_currents.keys()) == {"ika"}
    assert set(s.additional_gating.keys()) == {"a", "b"}


def test_from_dataframe_missing_classic_column_returns_empty_list() -> None:
    """When a classic column is absent the corresponding field is an empty list."""
    df = _make_df()
    df = df.drop(columns=["voltage"])
    s = Sweep.from_dataframe(df, _make_stimulus(), "", "", "Current Clamp")
    assert s.voltage == []


def test_from_dataframe_current_clamp_mode_stored() -> None:
    """clamp_mode is stored verbatim from the mode argument."""
    s = Sweep.from_dataframe(_make_df(), _make_stimulus(), "", "", "Current Clamp")
    assert s.clamp_mode == "Current Clamp"


def test_from_dataframe_voltage_clamp_mode_stored() -> None:
    """clamp_mode is stored correctly for Voltage Clamp."""
    s = Sweep.from_dataframe(_make_df(), _make_stimulus(), "", "", "Voltage Clamp")
    assert s.clamp_mode == "Voltage Clamp"


def test_from_dataframe_label_and_color_stored() -> None:
    """Label and color are stored verbatim."""
    s = Sweep.from_dataframe(_make_df(), _make_stimulus(), "My Label", "#abcdef", "CC")
    assert s.label == "My Label"
    assert s.color == "#abcdef"


# ---------------------------------------------------------------------------
# build_figure — return type and subplot structure
# ---------------------------------------------------------------------------


def _all_flags_true() -> dict:
    """Return keyword arguments for build_figure with all visibility flags True.

    Returns:
        Dict of flag arguments suitable for unpacking into build_figure.
    """
    return dict(
        show_voltage=True,
        show_total_current=True,
        show_sodium_current=True,
        show_potassium_current=True,
        show_leak_current=True,
        show_potassium_activation=True,
        show_sodium_activation=True,
        show_sodium_inactivation=True,
    )


def test_build_figure_returns_go_figure() -> None:
    """build_figure returns a plotly go.Figure."""
    fig = build_figure([], [], clamp_mode="Current Clamp", **_all_flags_true())
    assert isinstance(fig, go.Figure)


def test_build_figure_cc_has_three_subplots() -> None:
    """Current Clamp figure has exactly 3 subplots (rows)."""
    sweep = _make_sweep(mode="Current Clamp")
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **_all_flags_true())
    # Each subplot contributes a distinct y-axis entry (yaxis, yaxis2, yaxis3).
    yaxes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
    assert len(yaxes) == 3


def test_build_figure_vc_has_three_subplots() -> None:
    """Voltage Clamp figure has exactly 3 subplots."""
    sweep = _make_sweep(mode="Voltage Clamp")
    fig = build_figure([sweep], [], clamp_mode="Voltage Clamp", **_all_flags_true())
    yaxes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
    assert len(yaxes) == 3


def test_build_figure_empty_sweeps_no_error() -> None:
    """build_figure with no sweeps returns a valid empty figure."""
    fig = build_figure([], [], clamp_mode="Current Clamp", **_all_flags_true())
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# build_figure — single-sweep trace counts
# ---------------------------------------------------------------------------


def test_build_figure_cc_single_sweep_trace_count() -> None:
    """Current Clamp single sweep: voltage + 3 gating + stimulus = 5 traces."""
    sweep = _make_sweep(mode="Current Clamp")
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **_all_flags_true())
    # voltage(1) + n, m, h(3) + stimulus(1) = 5
    assert len(fig.data) == 5


def test_build_figure_vc_single_sweep_trace_count() -> None:
    """Voltage Clamp single sweep: 4 current traces + 3 gating + stimulus = 8."""
    sweep = _make_sweep(mode="Voltage Clamp")
    fig = build_figure([sweep], [], clamp_mode="Voltage Clamp", **_all_flags_true())
    # total, Na, K, leak(4) + n, m, h(3) + stimulus(1) = 8
    assert len(fig.data) == 8


# ---------------------------------------------------------------------------
# build_figure — hovermode
# ---------------------------------------------------------------------------


def test_build_figure_single_sweep_hovermode_x_unified() -> None:
    """Single-sweep mode uses hovermode='x unified'."""
    sweep = _make_sweep()
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **_all_flags_true())
    assert fig.layout.hovermode == "x unified"


def test_build_figure_multi_sweep_hovermode_x() -> None:
    """Multi-sweep (I-V Curve) mode uses hovermode='x'."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40, -20]]
    fig = build_figure(sweeps, [], clamp_mode="Voltage Clamp", **_all_flags_true())
    assert fig.layout.hovermode == "x"


# ---------------------------------------------------------------------------
# build_figure — multi-sweep carrier traces
# ---------------------------------------------------------------------------


def _count_carrier_traces(fig: go.Figure) -> int:
    """Count invisible carrier traces (showlegend=False, hovertemplate set).

    Args:
        fig: The Plotly figure to inspect.

    Returns:
        Number of carrier traces found.
    """
    count = 0
    for trace in fig.data:
        if (
            trace.showlegend is False
            and trace.hovertemplate is not None
            and "%{customdata}" in trace.hovertemplate
        ):
            count += 1
    return count


def test_build_figure_multi_sweep_adds_three_carrier_traces() -> None:
    """Multi-sweep mode adds exactly 3 carrier traces (one per subplot)."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40, -20]]
    fig = build_figure(sweeps, [], clamp_mode="Voltage Clamp", **_all_flags_true())
    assert _count_carrier_traces(fig) == 3


def test_build_figure_single_sweep_has_no_carrier_traces() -> None:
    """Single-sweep mode adds no carrier traces."""
    sweep = _make_sweep()
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **_all_flags_true())
    assert _count_carrier_traces(fig) == 0


def test_build_figure_multi_sweep_data_traces_hoverinfo_skip() -> None:
    """In multi-sweep mode, non-carrier traces have hoverinfo='skip'."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40]]
    fig = build_figure(sweeps, [], clamp_mode="Current Clamp", **_all_flags_true())
    carrier_count = _count_carrier_traces(fig)
    for trace in fig.data:
        is_carrier = (
            trace.showlegend is False
            and trace.hovertemplate is not None
            and "%{customdata}" in (trace.hovertemplate or "")
        )
        if not is_carrier:
            assert trace.hoverinfo == "skip", (
                f"Expected hoverinfo='skip' on non-carrier trace '{trace.name}'"
            )
    assert carrier_count == 3


# ---------------------------------------------------------------------------
# build_figure — visibility flags
# ---------------------------------------------------------------------------


def test_build_figure_hidden_voltage_trace_is_present_but_not_visible() -> None:
    """Voltage trace is in fig.data but has visible=False when toggled off."""
    sweep = _make_sweep(mode="Current Clamp")
    flags = _all_flags_true()
    flags["show_voltage"] = False
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **flags)
    voltage_traces = [t for t in fig.data if "Voltage" in (t.name or "")]
    assert len(voltage_traces) == 1
    assert voltage_traces[0].visible is False


def test_build_figure_hidden_trace_does_not_remove_it() -> None:
    """Disabling a flag sets visible=False; it never removes the trace."""
    sweep = _make_sweep(mode="Current Clamp")
    flags_on = _all_flags_true()
    flags_off = {**flags_on, "show_voltage": False}
    fig_on = build_figure([sweep], [], clamp_mode="Current Clamp", **flags_on)
    fig_off = build_figure([sweep], [], clamp_mode="Current Clamp", **flags_off)
    assert len(fig_on.data) == len(fig_off.data)


def test_build_figure_gating_traces_hidden_when_flags_off() -> None:
    """All gating traces are present but hidden when their flags are False."""
    # Use an empty label so trace names are plain "n", "m", "h".
    sweep = _make_sweep(label="", mode="Current Clamp")
    flags = _all_flags_true()
    flags["show_potassium_activation"] = False
    flags["show_sodium_activation"] = False
    flags["show_sodium_inactivation"] = False
    fig = build_figure([sweep], [], clamp_mode="Current Clamp", **flags)
    gating_traces = [t for t in fig.data if t.name in ("n", "m", "h")]
    assert len(gating_traces) == 3
    assert all(tr.visible is False for tr in gating_traces)


# ---------------------------------------------------------------------------
# build_figure — saved sweeps
# ---------------------------------------------------------------------------


def test_build_figure_saved_sweep_adds_traces() -> None:
    """A saved sweep adds traces on top of current sweep traces."""
    current = _make_sweep(label="current", color="#ff0000")
    saved = _make_sweep(label="saved", color="#888888")
    fig_no_saved = build_figure(
        [current], [], clamp_mode="Current Clamp", **_all_flags_true()
    )
    fig_with_saved = build_figure(
        [current], [saved], clamp_mode="Current Clamp", **_all_flags_true()
    )
    assert len(fig_with_saved.data) > len(fig_no_saved.data)


def test_build_figure_saved_sweep_stimulus_trace_present() -> None:
    """A saved sweep always includes a stimulus trace on the stimulus subplot."""
    current = _make_sweep(label="current")
    saved = _make_sweep(label="saved_ref", color="#666666")
    fig = build_figure(
        [current], [saved], clamp_mode="Current Clamp", **_all_flags_true()
    )
    saved_stimulus_traces = [t for t in fig.data if t.name == "saved_ref"]
    assert len(saved_stimulus_traces) == 1


# ---------------------------------------------------------------------------
# _build_hover_tables — return shape and length
# ---------------------------------------------------------------------------


def _default_hover_args(sweeps: list[Sweep], *, is_vc: bool = False) -> dict:
    """Build default keyword arguments for _build_hover_tables.

    Args:
        sweeps: List of Sweep instances to pass as current_sweeps.
        is_vc: Whether to use Voltage Clamp column layout.

    Returns:
        Dict of keyword arguments suitable for unpacking into _build_hover_tables.
    """
    return dict(
        current_sweeps=sweeps,
        show_total_current=True,
        show_sodium_current=True,
        show_potassium_current=True,
        show_leak_current=True,
        show_potassium_activation=True,
        show_sodium_activation=True,
        show_sodium_inactivation=True,
        show_additional_currents={},
        show_additional_gating={},
        add_current_keys=[],
        add_gating_keys=[],
        is_vc=is_vc,
        stride=1,
    )


def test_build_hover_tables_returns_three_lists() -> None:
    """_build_hover_tables returns a 3-tuple of lists."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40]]
    result = _build_hover_tables(**_default_hover_args(sweeps))
    assert len(result) == 3
    for lst in result:
        assert isinstance(lst, list)


def test_build_hover_tables_lists_have_equal_length() -> None:
    """All three returned lists have the same length."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40, -20]]
    resp, gating, stim = _build_hover_tables(**_default_hover_args(sweeps))
    assert len(resp) == len(gating) == len(stim)


def test_build_hover_tables_length_equals_time_points_at_stride_1() -> None:
    """With stride=1, output length equals the number of time points."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    resp, _, _ = _build_hover_tables(**_default_hover_args(sweeps))
    assert len(resp) == _N


def test_build_hover_tables_stride_downsamples_output() -> None:
    """With stride=k, output length equals ceil(n / k)."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    stride = 5
    args = {**_default_hover_args(sweeps), "stride": stride}
    resp, _, _ = _build_hover_tables(**args)
    expected = math.ceil(_N / stride)
    assert len(resp) == expected


def test_build_hover_tables_html_contains_sweep_label() -> None:
    """Each HTML string contains the sweep label."""
    sweeps = [_make_sweep(label="MySweep"), _make_sweep(label="OtherSweep")]
    resp, _, _ = _build_hover_tables(**_default_hover_args(sweeps))
    assert all("MySweep" in html for html in resp)
    assert all("OtherSweep" in html for html in resp)


def test_build_hover_tables_cc_resp_contains_voltage_header() -> None:
    """Current Clamp response HTML contains the voltage column header."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    resp, _, _ = _build_hover_tables(**_default_hover_args(sweeps, is_vc=False))
    assert all("V (mV)" in html for html in resp)


def test_build_hover_tables_vc_resp_contains_current_header() -> None:
    """Voltage Clamp response HTML contains a current column header."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    resp, _, _ = _build_hover_tables(**_default_hover_args(sweeps, is_vc=True))
    assert all("I_total" in html for html in resp)


def test_build_hover_tables_stim_html_contains_time_data() -> None:
    """Stimulus HTML strings are non-empty and contain numeric content."""
    sweeps = [_make_sweep(label="X"), _make_sweep(label="Y")]
    _, _, stim = _build_hover_tables(**_default_hover_args(sweeps))
    assert all(len(html) > 0 for html in stim)


def test_build_hover_tables_all_flags_off_gating_returns_empty_strings() -> None:
    """When all gating visibility flags are False, gating HTML entries are empty."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    args = {
        **_default_hover_args(sweeps),
        "show_potassium_activation": False,
        "show_sodium_activation": False,
        "show_sodium_inactivation": False,
    }
    _, gating, _ = _build_hover_tables(**args)
    assert all(html == "" for html in gating)


def test_build_hover_tables_vc_all_current_flags_off_resp_returns_empty_strings() -> (
    None
):
    """In VC mode with all current flags off, response HTML entries are empty."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    args = {
        **_default_hover_args(sweeps, is_vc=True),
        "show_total_current": False,
        "show_sodium_current": False,
        "show_potassium_current": False,
        "show_leak_current": False,
    }
    resp, _, _ = _build_hover_tables(**args)
    assert all(html == "" for html in resp)


def test_build_hover_tables_stride_2_length() -> None:
    """Stride 2 halves the number of hover points (ceiling division)."""
    n = 10
    sweeps = [_make_sweep(label="A", n=n), _make_sweep(label="B", n=n)]
    args = {**_default_hover_args(sweeps), "stride": 2}
    resp, _, _ = _build_hover_tables(**args)
    assert len(resp) == math.ceil(n / 2)
