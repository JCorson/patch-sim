"""Right-hand analysis sidebar component.

A collapsible panel with a tabbed layout.  Each tab hosts a different
analysis view; currently the only tab is "AP Metrics".  The tab structure
makes it straightforward to add further analysis views in future.
"""

import reflex as rx

from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.analysis import AnalysisState

_PANEL_WIDTH = "300px"
_COLLAPSED_WIDTH = "36px"


def _spike_row(spike: dict) -> rx.Component:
    """Render one row of the per-spike table.

    Args:
        spike: Dictionary with pre-formatted string values for each metric.

    Returns:
        A table row with per-spike metric cells.
    """
    return rx.table.row(
        rx.table.cell(rx.text(spike["index"].to(int) + 1, size="1")),
        rx.table.cell(rx.text(spike["threshold_voltage"].to(str), size="1")),
        rx.table.cell(rx.text(spike["peak_voltage"].to(str), size="1")),
        rx.table.cell(rx.text(spike["rise_time"].to(str), size="1")),
        rx.table.cell(rx.text(spike["half_width"].to(str), size="1")),
        rx.table.cell(rx.text(spike["ahp_depth"].to(str), size="1")),
    )


def _ap_summary() -> rx.Component:
    """Render the AP summary statistics section.

    Firing rate and mean ISI are omitted when the data is pooled from multiple
    sweeps (``AnalysisState.ap_is_multi_sweep``), because those values are
    shown per-sweep in the F-I curve.

    Returns:
        A compact grid of labelled metric values drawn from AnalysisState.ap_summary.
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


def _ap_fi_plot() -> rx.Component:
    """Render the embedded F-I curve inside the AP Metrics tab.

    Shown only when F-I data is available (current clamp multi-sweep).

    Returns:
        A compact Plotly F-I figure inside a flex container.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.fi_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _ap_metrics_tab() -> rx.Component:
    """Render the AP Metrics tab content.

    For single-sweep runs: shows summary statistics (including firing rate and
    ISI) and a per-spike detail table.  For multi-sweep current clamp runs:
    shows pooled AP summary (without firing rate / ISI), the F-I curve plot,
    and the pooled spike table.  When neither AP metrics nor F-I data are
    available, a placeholder message is shown instead.

    Returns:
        The full tab content as a flex column.
    """
    return rx.cond(
        AnalysisState.has_ap_or_fi,
        rx.flex(
            rx.cond(AnalysisState.has_ap_metrics, _ap_summary(), rx.box()),
            rx.cond(AnalysisState.has_fi_data, _ap_fi_plot(), rx.box()),
            rx.cond(AnalysisState.has_ap_metrics, _ap_spike_table(), rx.box()),
            direction="column",
            height="100%",
            width="100%",
            overflow="hidden",
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


def _gv_plot() -> rx.Component:
    """Render the g-V curve plot embedded below the I-V curve.

    Returns:
        A bordered flex container with the g-V Plotly figure.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.gv_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _iv_curve_tab() -> rx.Component:
    """Render the I-V Curve tab content.

    Shows a Plotly I-V curve with peak inward, peak outward, and steady-state
    traces when voltage clamp multi-sweep data is available.  When g-V data is
    also available, a normalised conductance plot with Boltzmann fit is shown
    below the I-V curve.  Displays a placeholder message when no data exists.

    Returns:
        The full tab content as a flex column.
    """
    return rx.cond(
        AnalysisState.has_iv_data,
        rx.scroll_area(
            rx.flex(
                rx.plotly(
                    data=AnalysisState.iv_figure,
                    width="100%",
                ),
                rx.cond(
                    AnalysisState.has_gv_data,
                    _gv_plot(),
                    rx.box(),
                ),
                direction="column",
                width="100%",
                padding="1",
            ),
            height="100%",
            width="100%",
        ),
        rx.flex(
            rx.text(
                "Run a voltage clamp multi-sweep simulation to see the I-V curve.",
                size="1",
                color="gray",
                text_align="center",
            ),
            padding="4",
            justify="center",
        ),
    )


def _expanded_panel() -> rx.Component:
    """Render the full expanded analysis sidebar.

    Returns:
        A fixed-width flex column with a header, tab strip, and tab content.
    """
    return rx.flex(
        # Header
        rx.hstack(
            rx.icon("chart-line", size=14),
            rx.text("Analysis", size="4", weight="bold"),
            rx.spacer(),
            rx.icon_button(
                rx.icon("panel-right-close", size=14),
                on_click=SimulationState.toggle_analysis_panel,
                variant="ghost",
                size="2",
                cursor="pointer",
            ),
            padding_x="3",
            padding_y="2",
            border_bottom="1px solid var(--gray-4)",
            width="100%",
            align="center",
        ),
        # Tabs
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("AP Metrics", value="ap", size="1"),
                rx.tabs.trigger("I-V Curve", value="iv", size="1"),
                padding_x="3",
                padding_y="1",
                border_bottom="1px solid var(--gray-4)",
            ),
            rx.tabs.content(
                _ap_metrics_tab(),
                value="ap",
                flex_grow="1",
                min_height="0",
                overflow="hidden",
                padding="0",
            ),
            rx.tabs.content(
                _iv_curve_tab(),
                value="iv",
                flex_grow="1",
                min_height="0",
                overflow="hidden",
                padding="0",
            ),
            default_value="ap",
            flex_grow="1",
            min_height="0",
            display="flex",
            flex_direction="column",
        ),
        direction="column",
        width=_PANEL_WIDTH,
        min_width=_PANEL_WIDTH,
        height="100%",
        overflow="hidden",
    )


def _collapsed_strip() -> rx.Component:
    """Render the narrow collapsed strip with a re-open button.

    Returns:
        A thin column containing just the expand icon button.
    """
    return rx.flex(
        rx.icon_button(
            rx.icon("panel-right-open", size=14),
            on_click=SimulationState.toggle_analysis_panel,
            variant="ghost",
            size="2",
            cursor="pointer",
        ),
        direction="column",
        align="center",
        padding_top="2",
        width=_COLLAPSED_WIDTH,
        min_width=_COLLAPSED_WIDTH,
        height="100%",
    )


def analysis_sidebar() -> rx.Component:
    """Render the right-hand analysis sidebar.

    Toggles between a full panel (with tabbed analysis views) and a thin
    collapsed strip.  The panel is toggled via SimulationState.toggle_analysis_panel.

    Returns:
        A Reflex component for the collapsible analysis sidebar.
    """
    return rx.box(
        rx.cond(
            SimulationState.analysis_panel_open,
            _expanded_panel(),
            _collapsed_strip(),
        ),
        border_left="1px solid var(--gray-4)",
        background="var(--gray-1)",
        height="calc(100vh - 60px - 2 * var(--space-3))",
        flex_shrink="0",
    )
