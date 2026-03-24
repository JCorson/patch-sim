"""Unit tests for patch_sim_ui/plotting.py.

Covers Sweep.from_dataframe, build_figure, and _build_hover_tables.
All three are pure functions with no Reflex dependency.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from patch_sim_ui.constants import CC_VOLTAGE_COLOR, STIMULUS_COLOR
from patch_sim_ui.plotting import (
    Sweep,
    TraceVisibility,
    _build_hover_tables,
    build_figure,
    compute_trace_visibility_map,
)

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
        "Na_current": list(np.zeros(n)),
        "K_current": list(np.zeros(n)),
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


def _all_flags_true() -> TraceVisibility:
    """Return a TraceVisibility with all classic flags set to True.

    Returns:
        A TraceVisibility instance with every flag enabled.
    """
    return TraceVisibility(
        voltage=True,
        total_current=True,
        sodium_current=True,
        potassium_current=True,
        leak_current=True,
        potassium_activation=True,
        sodium_activation=True,
        sodium_inactivation=True,
    )


def test_build_figure_returns_go_figure() -> None:
    """build_figure returns a plotly go.Figure."""
    fig = build_figure([], [], visibility=_all_flags_true(), clamp_mode="Current Clamp")
    assert isinstance(fig, go.Figure)


def test_build_figure_cc_has_three_subplots() -> None:
    """Current Clamp figure has exactly 3 subplots (rows)."""
    sweep = _make_sweep(mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    # Each subplot contributes a distinct y-axis entry (yaxis, yaxis2, yaxis3).
    yaxes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
    assert len(yaxes) == 3


def test_build_figure_vc_has_three_subplots() -> None:
    """Voltage Clamp figure has exactly 3 subplots."""
    sweep = _make_sweep(mode="Voltage Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    yaxes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
    assert len(yaxes) == 3


def test_build_figure_empty_sweeps_no_error() -> None:
    """build_figure with no sweeps returns a valid empty figure."""
    fig = build_figure([], [], visibility=_all_flags_true(), clamp_mode="Current Clamp")
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# build_figure — single-sweep trace counts
# ---------------------------------------------------------------------------


def test_build_figure_cc_single_sweep_trace_count() -> None:
    """Current Clamp single sweep: voltage + 3 gating + stimulus = 5 traces."""
    sweep = _make_sweep(mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    # voltage(1) + n, m, h(3) + stimulus(1) = 5
    assert len(fig.data) == 5


def test_build_figure_vc_single_sweep_trace_count() -> None:
    """Voltage Clamp single sweep: 4 current traces + 3 gating + stimulus = 8."""
    sweep = _make_sweep(mode="Voltage Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    # total, Na, K, leak(4) + n, m, h(3) + stimulus(1) = 8
    assert len(fig.data) == 8


# ---------------------------------------------------------------------------
# build_figure — CC voltage and stimulus/command trace colours
# ---------------------------------------------------------------------------


def test_build_figure_cc_voltage_uses_fixed_color() -> None:
    """Current Clamp voltage trace uses CC_VOLTAGE_COLOR, not the sweep color."""
    sweep = _make_sweep(label="", color="#ff0000", mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    voltage_traces = [t for t in fig.data if "Voltage" in (t.name or "")]
    assert len(voltage_traces) == 1, "Expected exactly one voltage trace"
    assert voltage_traces[0].line.color == CC_VOLTAGE_COLOR, (
        f"CC voltage should use CC_VOLTAGE_COLOR ({CC_VOLTAGE_COLOR!r}), "
        f"got {voltage_traces[0].line.color!r}"
    )


def test_build_figure_cc_stimulus_uses_stimulus_color() -> None:
    """Current Clamp stimulus trace uses STIMULUS_COLOR."""
    sweep = _make_sweep(label="", color="#ff0000", mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    stim_traces = [t for t in fig.data if "Stimulus" in (t.name or "")]
    assert len(stim_traces) == 1, "Expected exactly one stimulus trace"
    assert stim_traces[0].line.color == STIMULUS_COLOR, (
        f"CC stimulus should use STIMULUS_COLOR ({STIMULUS_COLOR!r}), "
        f"got {stim_traces[0].line.color!r}"
    )


def test_build_figure_vc_command_uses_stimulus_color() -> None:
    """Voltage Clamp command trace uses STIMULUS_COLOR, matching CC stimulus."""
    sweep = _make_sweep(label="", color="#ff0000", mode="Voltage Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    cmd_traces = [t for t in fig.data if "Command" in (t.name or "")]
    assert len(cmd_traces) == 1, "Expected exactly one Command trace"
    assert cmd_traces[0].line.color == STIMULUS_COLOR, (
        f"VC command should use STIMULUS_COLOR ({STIMULUS_COLOR!r}), "
        f"got {cmd_traces[0].line.color!r}"
    )


# ---------------------------------------------------------------------------
# build_figure — hovermode
# ---------------------------------------------------------------------------


def test_build_figure_single_sweep_hovermode_x_unified() -> None:
    """Single-sweep mode uses hovermode='x unified'."""
    sweep = _make_sweep()
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    assert fig.layout.hovermode == "x unified"


def test_build_figure_multi_sweep_hovermode_x() -> None:
    """Multi-sweep (I-V Curve) mode uses hovermode='x'."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40, -20]]
    fig = build_figure(
        sweeps, [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    assert fig.layout.hovermode == "x"


def test_build_figure_show_hover_false_disables_hovermode() -> None:
    """When show_hover is False the figure hovermode is set to False."""
    sweep = _make_sweep()
    fig = build_figure(
        [sweep],
        [],
        visibility=_all_flags_true(),
        clamp_mode="Current Clamp",
        show_hover=False,
    )
    assert fig.layout.hovermode is False


def test_build_figure_show_hover_false_multi_sweep() -> None:
    """When show_hover is False multi-sweep figures also disable hovermode."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40, -20]]
    fig = build_figure(
        sweeps,
        [],
        visibility=_all_flags_true(),
        clamp_mode="Voltage Clamp",
        show_hover=False,
    )
    assert fig.layout.hovermode is False


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
    fig = build_figure(
        sweeps, [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    assert _count_carrier_traces(fig) == 3


def test_build_figure_single_sweep_has_no_carrier_traces() -> None:
    """Single-sweep mode adds no carrier traces."""
    sweep = _make_sweep()
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    assert _count_carrier_traces(fig) == 0


def test_build_figure_multi_sweep_data_traces_hoverinfo_skip() -> None:
    """In multi-sweep mode, non-carrier traces have hoverinfo='skip'."""
    sweeps = [_make_sweep(label=f"{v} mV") for v in [-60, -40]]
    fig = build_figure(
        sweeps, [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
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
    vis = TraceVisibility(voltage=False)
    fig = build_figure([sweep], [], visibility=vis, clamp_mode="Current Clamp")
    voltage_traces = [t for t in fig.data if "Voltage" in (t.name or "")]
    assert len(voltage_traces) == 1
    assert voltage_traces[0].visible is False


def test_build_figure_hidden_trace_does_not_remove_it() -> None:
    """Disabling a flag sets visible=False; it never removes the trace."""
    sweep = _make_sweep(mode="Current Clamp")
    vis_on = TraceVisibility()
    vis_off = TraceVisibility(voltage=False)
    fig_on = build_figure([sweep], [], visibility=vis_on, clamp_mode="Current Clamp")
    fig_off = build_figure([sweep], [], visibility=vis_off, clamp_mode="Current Clamp")
    assert len(fig_on.data) == len(fig_off.data)


def test_build_figure_gating_traces_hidden_when_flags_off() -> None:
    """All gating traces are present but hidden when their flags are False."""
    # Use an empty label so trace names are plain "n", "m", "h".
    sweep = _make_sweep(label="", mode="Current Clamp")
    vis = TraceVisibility(
        potassium_activation=False,
        sodium_activation=False,
        sodium_inactivation=False,
    )
    fig = build_figure([sweep], [], visibility=vis, clamp_mode="Current Clamp")
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
        [current], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    fig_with_saved = build_figure(
        [current], [saved], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    assert len(fig_with_saved.data) > len(fig_no_saved.data)


def test_build_figure_saved_sweep_stimulus_trace_present() -> None:
    """A saved sweep always includes a stimulus trace on the stimulus subplot."""
    current = _make_sweep(label="current")
    saved = _make_sweep(label="saved_ref", color="#666666")
    fig = build_figure(
        [current], [saved], visibility=_all_flags_true(), clamp_mode="Current Clamp"
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
        visibility=TraceVisibility(),
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
    vis = TraceVisibility(
        potassium_activation=False,
        sodium_activation=False,
        sodium_inactivation=False,
    )
    args = {**_default_hover_args(sweeps), "visibility": vis}
    _, gating, _ = _build_hover_tables(**args)
    assert all(html == "" for html in gating)


def test_build_hover_tables_vc_all_current_flags_off_resp_returns_empty_strings() -> (
    None
):
    """In VC mode with all current flags off, response HTML entries are empty."""
    sweeps = [_make_sweep(label="A"), _make_sweep(label="B")]
    vis = TraceVisibility(
        total_current=False,
        sodium_current=False,
        potassium_current=False,
        leak_current=False,
    )
    args = {**_default_hover_args(sweeps, is_vc=True), "visibility": vis}
    resp, _, _ = _build_hover_tables(**args)
    assert all(html == "" for html in resp)


def test_build_hover_tables_stride_2_length() -> None:
    """Stride 2 halves the number of hover points (ceiling division)."""
    n = 10
    sweeps = [_make_sweep(label="A", n=n), _make_sweep(label="B", n=n)]
    args = {**_default_hover_args(sweeps), "stride": 2}
    resp, _, _ = _build_hover_tables(**args)
    assert len(resp) == math.ceil(n / 2)


# ---------------------------------------------------------------------------
# compute_trace_visibility_map
# ---------------------------------------------------------------------------


def test_compute_trace_visibility_map_cc_single_sweep_classic_fields() -> None:
    """CC single sweep maps classic show_* fields to the correct indices."""
    sweep = _make_sweep(mode="Current Clamp")
    result = compute_trace_visibility_map([sweep], [], "Current Clamp")
    # trace order: voltage(0), n(1), m(2), h(3), stimulus(4, not mapped)
    assert result["show_voltage"] == [0]
    assert result["show_potassium_activation"] == [1]
    assert result["show_sodium_activation"] == [2]
    assert result["show_sodium_inactivation"] == [3]
    assert "show_leak_current" not in result


def test_compute_trace_visibility_map_vc_single_sweep_classic_fields() -> None:
    """VC single sweep maps classic show_* fields to the correct indices."""
    sweep = _make_sweep(mode="Voltage Clamp")
    result = compute_trace_visibility_map([sweep], [], "Voltage Clamp")
    # trace order: total(0), Na(1), K(2), leak(3), n(4), m(5), h(6), stim(7)
    assert result["show_total_current"] == [0]
    assert result["show_sodium_current"] == [1]
    assert result["show_potassium_current"] == [2]
    assert result["show_leak_current"] == [3]
    assert result["show_potassium_activation"] == [4]
    assert result["show_sodium_activation"] == [5]
    assert result["show_sodium_inactivation"] == [6]
    assert "show_voltage" not in result


def test_compute_trace_visibility_map_cc_multi_sweep_accumulates_indices() -> None:
    """Multi-sweep CC maps each field to one index per sweep."""
    sweeps = [_make_sweep(mode="Current Clamp"), _make_sweep(mode="Current Clamp")]
    result = compute_trace_visibility_map(sweeps, [], "Current Clamp")
    # Sweep 0: voltage(0), n(1), m(2), h(3), stim(4)
    # Sweep 1: voltage(5), n(6), m(7), h(8), stim(9)
    assert result["show_voltage"] == [0, 5]
    assert result["show_potassium_activation"] == [1, 6]
    assert result["show_sodium_activation"] == [2, 7]
    assert result["show_sodium_inactivation"] == [3, 8]


def test_compute_trace_visibility_map_saved_sweep_does_not_shift_current_indices() -> (
    None
):
    """Adding a saved sweep does not change current-sweep trace indices."""
    current = _make_sweep(mode="Current Clamp")
    saved = _make_sweep(mode="Current Clamp")
    result_without = compute_trace_visibility_map([current], [], "Current Clamp")
    result_with = compute_trace_visibility_map([current], [saved], "Current Clamp")
    assert result_without == result_with


def test_compute_trace_visibility_map_cc_additional_gating_mapped() -> None:
    """CC sweep with additional gating maps the field to the correct index."""
    extra = {"r": [0.0] * _N}
    sweep = _make_sweep(mode="Current Clamp", extra_cols=extra)
    gating_map = {"r": "show_ih_gating"}
    result = compute_trace_visibility_map(
        [sweep], [], "Current Clamp", additional_gating_field_map=gating_map
    )
    # voltage(0), n(1), m(2), h(3), r(4), stim(5)
    assert result["show_ih_gating"] == [4]
    assert result["show_sodium_inactivation"] == [3]


def test_compute_trace_visibility_map_vc_additional_current_mapped() -> None:
    """VC sweep with additional current maps the field to the correct index."""
    extra = {"foo_current": [0.0] * _N}
    sweep = _make_sweep(mode="Voltage Clamp", extra_cols=extra)
    curr_map = {"foo": "show_foo_current"}
    result = compute_trace_visibility_map(
        [sweep], [], "Voltage Clamp", additional_current_field_map=curr_map
    )
    # total(0), Na(1), K(2), leak(3), foo(4), n(5), m(6), h(7), stim(8)
    assert result["show_foo_current"] == [4]
    assert result["show_potassium_activation"] == [5]
    assert result["show_sodium_activation"] == [6]
    assert result["show_sodium_inactivation"] == [7]


def test_compute_trace_visibility_map_multi_gating_keys_same_field() -> None:
    """Two gating keys sharing one field name both map to that field."""
    extra = {"a": [0.0] * _N, "b": [0.0] * _N}
    sweep = _make_sweep(mode="Current Clamp", extra_cols=extra)
    gating_map = {"a": "show_ika_gating", "b": "show_ika_gating"}
    result = compute_trace_visibility_map(
        [sweep], [], "Current Clamp", additional_gating_field_map=gating_map
    )
    # voltage(0), n(1), m(2), h(3), a(4), b(5), stim(6)
    assert result["show_ika_gating"] == [4, 5]


def test_compute_trace_visibility_map_unknown_additional_key_advances_counter() -> None:
    """Additional keys absent from the field map still advance the index counter."""
    extra = {"unknown_current": [0.0] * _N}
    sweep = _make_sweep(mode="Voltage Clamp", extra_cols=extra)
    result = compute_trace_visibility_map([sweep], [], "Voltage Clamp")
    # unknown advances the counter; classic gating should be at 5, 6, 7
    assert result["show_potassium_activation"] == [5]
    assert result["show_sodium_activation"] == [6]
    assert result["show_sodium_inactivation"] == [7]


# ---------------------------------------------------------------------------
# build_figure with stored_traces
# ---------------------------------------------------------------------------


def test_build_figure_with_stored_traces_does_not_raise() -> None:
    """build_figure with non-empty stored_traces returns a Figure without error."""
    sweep = _make_sweep(mode="Current Clamp")
    stored = _make_sweep(mode="Current Clamp")
    stored = stored.model_copy(update={"label": "Stored 1"})
    fig = build_figure(
        [sweep], [], TraceVisibility(), "Current Clamp", stored_traces=[stored]
    )
    assert isinstance(fig, go.Figure)


def test_build_figure_stored_traces_adds_extra_traces() -> None:
    """build_figure with stored traces produces more traces than without."""
    sweep = _make_sweep(mode="Current Clamp")
    stored = _make_sweep(mode="Current Clamp")

    fig_no_stored = build_figure([sweep], [], TraceVisibility(), "Current Clamp")
    fig_with_stored = build_figure(
        [sweep], [], TraceVisibility(), "Current Clamp", stored_traces=[stored]
    )

    assert len(fig_with_stored.data) > len(fig_no_stored.data)


def test_compute_trace_visibility_map_with_stored_traces_advances_counter() -> None:
    """Stored traces advance trace index counter without polluting the result map."""
    sweep = _make_sweep(mode="Current Clamp")
    stored = _make_sweep(mode="Current Clamp")

    result_without = compute_trace_visibility_map([sweep], [], "Current Clamp")
    result_with = compute_trace_visibility_map(
        [sweep], [], "Current Clamp", stored_traces=[stored]
    )

    # The recorded show_* indices must be identical regardless of stored traces.
    assert result_without == result_with


# ---------------------------------------------------------------------------
# build_figure — legend visibility
# ---------------------------------------------------------------------------


def test_build_figure_cc_single_sweep_stimulus_not_in_legend() -> None:
    """Current Clamp single sweep: stimulus trace is excluded from the legend."""
    sweep = _make_sweep(mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    stim_traces = [t for t in fig.data if "Stimulus" in (t.name or "")]
    assert all(t.showlegend is False for t in stim_traces), (
        "Stimulus traces must not appear in the legend"
    )


def test_build_figure_vc_single_sweep_command_not_in_legend() -> None:
    """Voltage Clamp single sweep: command trace is excluded from the legend."""
    sweep = _make_sweep(mode="Voltage Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    cmd_traces = [t for t in fig.data if "Command" in (t.name or "")]
    assert all(t.showlegend is False for t in cmd_traces), (
        "Command traces must not appear in the legend"
    )


def test_build_figure_cc_single_sweep_only_gating_in_legend() -> None:
    """Current Clamp single sweep: voltage is suppressed (sole trace); gating shown."""
    sweep = _make_sweep(label="", mode="Current Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    legend_names = {t.name for t in fig.data if t.showlegend is not False}
    # Row 1 has only Voltage — suppressed because sole entry.
    assert "Voltage (mV)" not in legend_names
    # Gating row has 3 entries so all three are shown.
    assert "n" in legend_names
    assert "m" in legend_names
    assert "h" in legend_names


def test_build_figure_vc_single_sweep_currents_and_gating_in_legend() -> None:
    """Voltage Clamp single sweep: current and gating traces appear in the legend."""
    sweep = _make_sweep(label="", mode="Voltage Clamp")
    fig = build_figure(
        [sweep], [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    legend_traces = [t for t in fig.data if t.showlegend is not False]
    names = {t.name for t in legend_traces}
    assert "I_total" in names
    assert "I_Na" in names
    assert "I_K" in names
    assert "I_L" in names
    assert "n" in names
    assert "m" in names
    assert "h" in names


def test_build_figure_multi_sweep_only_first_sweep_in_legend() -> None:
    """Multi-sweep mode: only the first sweep's traces appear in the legend."""
    sweeps = [_make_sweep(label=f"{v} mV", mode="Voltage Clamp") for v in [-60, -40]]
    fig = build_figure(
        sweeps, [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    legend_names = {t.name for t in fig.data if t.showlegend is not False}
    # Traces from the second sweep must not appear.
    assert not any(n.startswith("-40 mV ") for n in legend_names), (
        "Second sweep traces must not appear in the legend"
    )


def test_build_figure_vc_multi_sweep_legend_names_have_no_voltage_prefix() -> None:
    """VC multi-sweep: legend entries use bare channel names, no command level."""
    sweeps = [_make_sweep(label=f"{v} mV", mode="Voltage Clamp") for v in [-60, -40]]
    fig = build_figure(
        sweeps, [], visibility=_all_flags_true(), clamp_mode="Voltage Clamp"
    )
    legend_names = {t.name for t in fig.data if t.showlegend is not False}
    # Current traces must use bare channel labels, not "-60 mV I_total" etc.
    assert "I_total" in legend_names
    assert "I_Na" in legend_names
    assert "I_K" in legend_names
    assert "I_L" in legend_names
    channel_names = {"I_total", "I_Na", "I_K", "I_L"}
    assert not any("mV" in n for n in legend_names if n in channel_names)


def test_build_figure_saved_sweep_stimulus_not_in_legend() -> None:
    """Saved sweep stimulus traces are excluded from the legend."""
    current = _make_sweep(mode="Current Clamp")
    saved = _make_sweep(label="saved", color="#aabbcc", mode="Current Clamp")
    fig = build_figure(
        [current], [saved], visibility=_all_flags_true(), clamp_mode="Current Clamp"
    )
    # The saved sweep's stimulus trace uses sweep.label ("saved") as its name.
    saved_stim = [t for t in fig.data if t.name == "saved" and t.showlegend is False]
    assert len(saved_stim) >= 1, "Saved sweep stimulus must be excluded from the legend"


def test_build_figure_stored_trace_stimulus_not_in_legend() -> None:
    """Stored trace stimulus traces are excluded from the legend."""
    sweep = _make_sweep(mode="Current Clamp")
    stored = _make_sweep(mode="Current Clamp")
    stored = stored.model_copy(update={"label": "Ref"})
    fig = build_figure(
        [sweep], [], TraceVisibility(), "Current Clamp", stored_traces=[stored]
    )
    # Stored traces add two traces named "Ref": one on row 1, one on stimulus row.
    ref_traces = [t for t in fig.data if t.name == "Ref"]
    assert len(ref_traces) == 2  # noqa: PLR2004
    stimulus_ref = [t for t in ref_traces if t.showlegend is False]
    assert len(stimulus_ref) == 1, "Stored trace stimulus must be excluded from legend"
