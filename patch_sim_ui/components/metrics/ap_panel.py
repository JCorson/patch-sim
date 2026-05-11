"""Action-potential metrics panel: summary, per-spike table, sub-tab composer.

Hosts the AP-specific renderers (summary grid, per-spike detail table, phase
plane plot) and the multi-tab composer that binds the AP, Bursts, and
Calcium views together for the current-clamp pane.
"""

import reflex as rx

from patch_sim_ui.components.metrics._row import RowColumn, metrics_row
from patch_sim_ui.components.metrics.burst_panel import (
    _ap_bursts_tab,
    _burst_summary,
)
from patch_sim_ui.components.metrics.calcium_panel import (
    _ap_calcium_tab,
    _ca_transient_summary,
)
from patch_sim_ui.components.metrics.fi_gv_panel import _ap_fi_plot
from patch_sim_ui.components.metrics.sfa_hyperpol_panel import (
    _ap_sfa_plot,
    _hyperpolarization_plot,
)
from patch_sim_ui.state.analysis import AnalysisState

_SPIKE_COLUMNS: tuple[RowColumn, ...] = (
    RowColumn("index", kind="int_1based"),
    RowColumn("threshold_voltage"),
    RowColumn("peak_voltage"),
    RowColumn("rise_time"),
    RowColumn("half_width"),
    RowColumn("ahp_depth"),
)


def _spike_row(spike: dict) -> rx.Component:
    """Render one row of the per-spike table.

    Args:
        spike: Dictionary with pre-formatted string values for each metric.

    Returns:
        A table row with per-spike metric cells.
    """
    return metrics_row(spike, _SPIKE_COLUMNS)


def _ap_summary() -> rx.Component:
    """Render the AP summary statistics section.

    Firing rate and mean ISI are omitted when the data is pooled from multiple
    sweeps (``AnalysisState.ap_is_multi_sweep``), because those values are
    shown per-sweep in the F-I curve.

    Returns:
        A compact grid of labeled metric values drawn from AnalysisState.ap_summary.
    """
    s = AnalysisState.ap_summary
    return rx.box(
        rx.grid(
            rx.text("Spikes", size="1", color="gray"),
            rx.text(s["spike_count"].to(str), size="1", align="left"),
            rx.text("Threshold", size="1", color="gray"),
            rx.text(
                s["mean_threshold_voltage"].to(str) + " mV", size="1", align="left"
            ),
            rx.text("Peak", size="1", color="gray"),
            rx.text(s["mean_peak_voltage"].to(str) + " mV", size="1", align="left"),
            rx.text("Rise time", size="1", color="gray"),
            rx.text(s["mean_rise_time"].to(str) + " ms", size="1", align="left"),
            rx.text("Half-width", size="1", color="gray"),
            rx.text(s["mean_half_width"].to(str) + " ms", size="1", align="left"),
            rx.text("AHP depth", size="1", color="gray"),
            rx.text(s["mean_ahp_depth"].to(str) + " mV", size="1", align="left"),
            rx.text("Rheobase", size="1", color="gray"),
            rx.text(s["rheobase"].to(str) + " µA/cm²", size="1", align="left"),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text("ISI", size="1", color="gray"),
                rx.box(),
            ),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text(s["mean_isi"].to(str) + " ms", size="1", align="left"),
                rx.box(),
            ),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text("Firing rate", size="1", color="gray"),
                rx.box(),
            ),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text(s["firing_rate"].to(str) + " Hz", size="1", align="left"),
                rx.box(),
            ),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text("Adapt. index", size="1", color="gray"),
                rx.box(),
            ),
            rx.cond(
                ~AnalysisState.ap_is_multi_sweep,
                rx.text(s["adaptation_index"].to(str), size="1", align="left"),
                rx.box(),
            ),
            columns="2",
            spacing="2",
            width="100%",
        ),
        padding="3",
        border_bottom="1px solid var(--gray-4)",
        width="100%",
    )


def _ap_spike_table() -> rx.Component:
    """Render the scrollable per-spike detail table.

    Returns:
        A flex-growing scroll area containing the per-spike metrics table.
    """
    return rx.scroll_area(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell(rx.text("#", size="1")),
                    rx.table.column_header_cell(rx.text("Thr (mV)", size="1")),
                    rx.table.column_header_cell(rx.text("Peak (mV)", size="1")),
                    rx.table.column_header_cell(rx.text("Rise (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("HW (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("AHP (mV)", size="1")),
                ),
            ),
            rx.table.body(
                rx.foreach(AnalysisState.ap_metrics, _spike_row),
            ),
            size="1",
            variant="surface",
            width="100%",
        ),
        padding="3",
        flex_grow="1",
        min_height="0",
    )


def _ap_phase_plane_plot() -> rx.Component:
    """Render the V vs dV/dt phase-plane plot inside the Analysis tab.

    Shows the membrane voltage on the x-axis against its time derivative on
    the y-axis.  One trajectory is drawn per sweep; multi-sweep runs display
    trajectories in their respective sweep colors.

    Returns:
        A flex container holding the phase-plane Plotly figure.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.phase_plane_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _ap_analysis_tab() -> rx.Component:
    """Render the Analysis sub-tab within the CC pane.

    Shows AP summary statistics, the burst summary (when ≥1 spike was
    detected), the calcium-transient summary (when calcium dynamics were
    active), the F-I curve (multi-sweep), the SFA curve, and the V vs
    dV/dt phase-plane when data are available.  Displays a placeholder
    when no data exists yet.

    Returns:
        The analysis sub-tab content as a scrollable flex column.
    """
    return rx.cond(
        AnalysisState.has_ap_or_fi
        | AnalysisState.has_ca_transient_metrics
        | AnalysisState.has_burst_summary,
        rx.scroll_area(
            rx.flex(
                rx.cond(AnalysisState.has_ap_metrics, _ap_summary(), rx.box()),
                rx.cond(
                    AnalysisState.has_burst_summary,
                    _burst_summary(),
                    rx.box(),
                ),
                rx.cond(
                    AnalysisState.has_ca_transient_metrics,
                    _ca_transient_summary(),
                    rx.box(),
                ),
                rx.cond(AnalysisState.has_fi_data, _ap_fi_plot(), rx.box()),
                rx.cond(
                    AnalysisState.has_hyperpolarization_data,
                    _hyperpolarization_plot(),
                    rx.box(),
                ),
                rx.cond(AnalysisState.has_sfa_data, _ap_sfa_plot(), rx.box()),
                rx.cond(
                    AnalysisState.has_phase_plane_data,
                    _ap_phase_plane_plot(),
                    rx.box(),
                ),
                direction="column",
                width="100%",
            ),
            height="100%",
            width="100%",
        ),
        rx.flex(
            rx.text(
                "Run a current clamp simulation to see AP metrics.",
                size="1",
                color="gray",
                text_align="center",
            ),
            padding="4",
            justify="center",
        ),
    )


def _ap_spikes_tab() -> rx.Component:
    """Render the Spikes sub-tab showing the per-spike detail table.

    Returns:
        The per-spike table wrapped in a flex column, or a placeholder when
        no spike data is available.
    """
    return rx.cond(
        AnalysisState.has_ap_metrics,
        _ap_spike_table(),
        rx.flex(
            rx.text(
                "No spikes detected.",
                size="1",
                color="gray",
                text_align="center",
            ),
            padding="4",
            justify="center",
        ),
    )


def _ap_metrics_tab() -> rx.Component:
    """Render the full AP Metrics panel with three sub-tabs.

    Sub-tabs:
    - **Analysis**: AP summary statistics, burst summary, calcium-transient
      summary, F-I curve, SFA curve, phase plane.
    - **Spikes**: Per-spike detail table.
    - **Bursts**: Per-burst detail table.
    - **Calcium**: Per-transient detail table (only relevant when calcium
      dynamics are active).

    Returns:
        A tabbed flex column for the CC analysis pane.
    """
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Analysis", value="analysis", size="1"),
            rx.tabs.trigger("Spikes", value="spikes", size="1"),
            rx.tabs.trigger("Bursts", value="bursts", size="1"),
            rx.tabs.trigger("Calcium", value="calcium", size="1"),
            size="1",
            width="100%",
        ),
        rx.tabs.content(
            _ap_analysis_tab(),
            value="analysis",
            height="100%",
            overflow="hidden",
        ),
        rx.tabs.content(
            _ap_spikes_tab(),
            value="spikes",
            height="100%",
            overflow="hidden",
        ),
        rx.tabs.content(
            _ap_bursts_tab(),
            value="bursts",
            height="100%",
            overflow="hidden",
        ),
        rx.tabs.content(
            _ap_calcium_tab(),
            value="calcium",
            height="100%",
            overflow="hidden",
        ),
        default_value="analysis",
        orientation="horizontal",
        height="100%",
        width="100%",
        display="flex",
        flex_direction="column",
        overflow="hidden",
    )
