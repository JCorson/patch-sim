"""Calcium-transient panel: summary, per-transient table, and tab content.

Renders the calcium-transient analysis views shown in the analysis sidebar
(both within the AP Metrics tab and on the I-V curve tab when calcium
dynamics were active).
"""

import reflex as rx

from patch_sim_ui.components.metrics._row import RowColumn, metrics_row
from patch_sim_ui.state.analysis import AnalysisState

_CA_TRANSIENT_COLUMNS: tuple[RowColumn, ...] = (
    RowColumn("index", kind="int_1based"),
    RowColumn("sweep_index", kind="gated_int", empty_text=""),
    RowColumn("peak_time"),
    RowColumn("peak_concentration"),
    RowColumn("time_to_peak"),
    RowColumn("decay_tau"),
    RowColumn("amplitude"),
)


def _ca_transient_row(transient: dict) -> rx.Component:
    """Render one row of the per-transient calcium-metric table.

    The ``decay_tau`` cell carries a trailing ``*`` when the τ value came
    from the 1/e fallback rather than a converged exponential fit; the
    ``sweep_index`` column shows the sweep number for multi-sweep runs (it
    is 0 for single-sweep results, displayed as a blank cell).

    Args:
        transient: Dictionary with pre-formatted string values for each metric.

    Returns:
        A table row with per-transient metric cells.
    """
    return metrics_row(transient, _CA_TRANSIENT_COLUMNS)


def _ca_transient_summary() -> rx.Component:
    """Render the calcium-transient summary statistics section.

    Returns:
        A compact grid of labelled metric values from
        AnalysisState.ca_transient_summary.  All concentrations are in µM
        and all times in ms.
    """
    s = AnalysisState.ca_transient_summary
    return rx.box(
        rx.heading("Calcium transients", size="1", margin_bottom="2"),
        rx.grid(
            rx.text("Transients", size="1", color="gray"),
            rx.text(s["transient_count"].to(str), size="1", align="left"),
            rx.text("Baseline", size="1", color="gray"),
            rx.text(
                s["baseline_concentration"].to(str) + " µM", size="1", align="left"
            ),
            rx.text("Peak", size="1", color="gray"),
            rx.text(
                s["mean_peak_concentration"].to(str) + " µM", size="1", align="left"
            ),
            rx.text("Time-to-peak", size="1", color="gray"),
            rx.text(s["mean_time_to_peak"].to(str) + " ms", size="1", align="left"),
            rx.text("Decay τ", size="1", color="gray"),
            rx.text(s["mean_decay_tau"].to(str) + " ms", size="1", align="left"),
            rx.text("Amplitude", size="1", color="gray"),
            rx.text(s["mean_amplitude"].to(str) + " µM", size="1", align="left"),
            columns="2",
            spacing="2",
            width="100%",
        ),
        padding="3",
        border_bottom="1px solid var(--gray-4)",
        width="100%",
    )


def _ca_transient_table() -> rx.Component:
    """Render the scrollable per-transient calcium detail table.

    Returns:
        A flex-growing scroll area containing the per-transient metrics table.
    """
    return rx.scroll_area(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell(rx.text("#", size="1")),
                    rx.table.column_header_cell(rx.text("Sweep", size="1")),
                    rx.table.column_header_cell(rx.text("Peak t (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("Peak (µM)", size="1")),
                    rx.table.column_header_cell(rx.text("TTP (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("τ (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("Δ[Ca] (µM)", size="1")),
                ),
            ),
            rx.table.body(
                rx.foreach(AnalysisState.ca_transient_metrics, _ca_transient_row),
            ),
            size="1",
            variant="surface",
            width="100%",
        ),
        padding="3",
        flex_grow="1",
        min_height="0",
    )


def _ap_calcium_tab() -> rx.Component:
    """Render the Calcium sub-tab showing the per-transient detail table.

    Returns:
        The per-transient table wrapped in a flex column, or a placeholder
        when calcium dynamics were not active or no transients were detected.
    """
    return rx.cond(
        AnalysisState.has_ca_transient_metrics,
        _ca_transient_table(),
        rx.flex(
            rx.text(
                "No calcium transients detected.",
                size="1",
                color="gray",
                text_align="center",
            ),
            padding="4",
            justify="center",
        ),
    )
