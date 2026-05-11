"""Burst-analysis panel: summary, per-burst table, and tab content."""

import reflex as rx

from patch_sim_ui.components.metrics._row import RowColumn, metrics_row
from patch_sim_ui.state.analysis import AnalysisState

# The Sweep cell uses a non-breaking space (U+00A0) for the empty branch
# to keep row baseline aligned across the table grid when this cell would
# otherwise be empty — a regular space collapses under default HTML
# white-space rules.
_BURST_COLUMNS: tuple[RowColumn, ...] = (
    RowColumn("index", kind="int_1based"),
    RowColumn("sweep_index", kind="gated_int", empty_text=" "),
    RowColumn("start_time"),
    RowColumn("end_time"),
    RowColumn("duration"),
    RowColumn("spike_count"),
    RowColumn("intra_burst_frequency"),
    RowColumn("mean_intra_burst_isi"),
)


def _burst_row(burst: dict) -> rx.Component:
    """Render one row of the per-burst detail table.

    The ``Sweep`` cell shows the 1-based sweep number for multi-sweep
    runs; for single-sweep runs the underlying ``sweep_index`` is 0 and
    the cell is blank.

    Args:
        burst: Dictionary with pre-formatted string values for each metric.

    Returns:
        A table row with per-burst metric cells.
    """
    return metrics_row(burst, _BURST_COLUMNS)


def _burst_summary() -> rx.Component:
    """Render the burst-analysis summary statistics section.

    Reads from :attr:`AnalysisState.burst_summary`.  Frequencies are in
    Hz, durations and intervals in ms, and the duty cycle is shown as a
    fraction of the recording window.  The applied ISI threshold and how
    it was chosen are surfaced so the user can interpret the result.

    Returns:
        A compact grid of labeled metric values.
    """
    s = AnalysisState.burst_summary
    return rx.box(
        rx.heading("Bursts", size="1", margin_bottom="2"),
        rx.grid(
            rx.text("Bursts", size="1", color="gray"),
            rx.text(s["burst_count"].to(str), size="1", align="left"),
            rx.text("Spikes/burst", size="1", color="gray"),
            rx.text(s["mean_spikes_per_burst"].to(str), size="1", align="left"),
            rx.text("Intra freq", size="1", color="gray"),
            rx.text(
                s["mean_intra_burst_frequency"].to(str) + " Hz",
                size="1",
                align="left",
            ),
            rx.text("IBI", size="1", color="gray"),
            rx.text(
                s["mean_inter_burst_interval"].to(str) + " ms", size="1", align="left"
            ),
            rx.text("Duty cycle", size="1", color="gray"),
            rx.text(s["duty_cycle"].to(str), size="1", align="left"),
            rx.text("ISI threshold", size="1", color="gray"),
            rx.text(s["isi_threshold_ms"].to(str) + " ms", size="1", align="left"),
            rx.text("Method", size="1", color="gray"),
            rx.text(s["threshold_method"].to(str), size="1", align="left"),
            columns="2",
            spacing="2",
            width="100%",
        ),
        padding="3",
        border_bottom="1px solid var(--gray-4)",
        width="100%",
    )


def _burst_table() -> rx.Component:
    """Render the scrollable per-burst detail table.

    Returns:
        A flex-growing scroll area containing the per-burst metrics table.
    """
    return rx.scroll_area(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell(rx.text("#", size="1")),
                    rx.table.column_header_cell(rx.text("Sweep", size="1")),
                    rx.table.column_header_cell(rx.text("Start (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("End (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("Dur (ms)", size="1")),
                    rx.table.column_header_cell(rx.text("Spikes", size="1")),
                    rx.table.column_header_cell(rx.text("Freq (Hz)", size="1")),
                    rx.table.column_header_cell(rx.text("ISI (ms)", size="1")),
                ),
            ),
            rx.table.body(
                rx.foreach(AnalysisState.burst_metrics, _burst_row),
            ),
            size="1",
            variant="surface",
            width="100%",
        ),
        padding="3",
        flex_grow="1",
        min_height="0",
    )


def _ap_bursts_tab() -> rx.Component:
    """Render the Bursts sub-tab showing the per-burst detail table.

    Returns:
        The per-burst table wrapped in a flex column, or a placeholder when
        no bursts were detected (and no spike-train was long enough to run
        the burst analyzer).
    """
    return rx.cond(
        AnalysisState.has_burst_summary,
        rx.cond(
            AnalysisState.burst_metrics.length() > 0,
            _burst_table(),
            rx.flex(
                rx.text(
                    "No bursts detected.",
                    size="1",
                    color="gray",
                    text_align="center",
                ),
                padding="4",
                justify="center",
            ),
        ),
        rx.flex(
            rx.text(
                "Burst analysis requires at least two spikes.",
                size="1",
                color="gray",
                text_align="center",
            ),
            padding="4",
            justify="center",
        ),
    )
