"""AP analysis metrics panel component.

Displays action potential summary statistics and per-spike metrics extracted
from the most recent current clamp simulation.  The panel is hidden when no
metrics are available (e.g. after a voltage clamp run or before the first
simulation).
"""

import reflex as rx

from patch_sim_ui.state import AppState


def _summary_row(label: str, value: rx.Var, unit: str = "") -> rx.Component:
    """Render one row of the summary table.

    Args:
        label: Metric name shown in the left column.
        value: Reactive var holding the formatted value string.
        unit: Optional unit string appended after the value.

    Returns:
        An hstack with the label and formatted value.
    """
    return rx.hstack(
        rx.text(label, size="1", color="gray", min_width="10em"),
        rx.text(value, size="1"),
        rx.text(unit, size="1", color="gray"),
        spacing="2",
        align="center",
    )


def _fmt(val: float | None, decimals: int = 2) -> str:
    """Format a nullable float for display.

    Args:
        val: The value to format, or None.
        decimals: Number of decimal places.

    Returns:
        A formatted string, or ``"—"`` when ``val`` is None.
    """
    if val is None:
        return "\u2014"
    return f"{val:.{decimals}f}"


def _spike_row(spike: dict) -> rx.Component:
    """Render one row of the per-spike table.

    Args:
        spike: Dictionary produced by ``dataclasses.asdict(SpikeMetrics)``.

    Returns:
        A table row with per-spike metric cells.
    """
    ahp = spike["ahp_depth"]
    ahp_str = rx.cond(
        ahp is None,
        "\u2014",
        ahp.to_string(),
    )
    return rx.table.row(
        rx.table.cell(
            rx.text(spike["index"].to(int) + 1, size="1"),
        ),
        rx.table.cell(
            rx.text(spike["threshold_voltage"].to(float).to_string(), size="1"),
        ),
        rx.table.cell(
            rx.text(spike["peak_voltage"].to(float).to_string(), size="1"),
        ),
        rx.table.cell(
            rx.text(spike["rise_time"].to(float).to_string(), size="1"),
        ),
        rx.table.cell(
            rx.text(spike["half_width"].to(float).to_string(), size="1"),
        ),
        rx.table.cell(
            rx.text(ahp_str, size="1"),
        ),
    )


def _summary_section() -> rx.Component:
    """Render the AP summary statistics grid.

    Returns:
        A vstack of labelled metric rows drawn from AppState.ap_summary.
    """
    s = AppState.ap_summary
    return rx.vstack(
        rx.text("Summary", size="1", weight="bold"),
        rx.grid(
            rx.text("Spikes detected", size="1", color="gray"),
            rx.text(s["spike_count"].to(int).to_string(), size="1"),
            rx.text("Mean threshold", size="1", color="gray"),
            rx.text(s["mean_threshold_voltage"].to_string(), size="1"),
            rx.text("mV", size="1", color="gray"),
            rx.text("Mean peak", size="1", color="gray"),
            rx.text(s["mean_peak_voltage"].to_string(), size="1"),
            rx.text("mV", size="1", color="gray"),
            rx.text("Mean rise time", size="1", color="gray"),
            rx.text(s["mean_rise_time"].to_string(), size="1"),
            rx.text("ms", size="1", color="gray"),
            rx.text("Mean half-width", size="1", color="gray"),
            rx.text(s["mean_half_width"].to_string(), size="1"),
            rx.text("ms", size="1", color="gray"),
            rx.text("Mean AHP depth", size="1", color="gray"),
            rx.text(s["mean_ahp_depth"].to_string(), size="1"),
            rx.text("mV", size="1", color="gray"),
            rx.text("Mean ISI", size="1", color="gray"),
            rx.text(s["mean_isi"].to_string(), size="1"),
            rx.text("ms", size="1", color="gray"),
            rx.text("Firing rate", size="1", color="gray"),
            rx.text(s["firing_rate"].to_string(), size="1"),
            rx.text("Hz", size="1", color="gray"),
            columns="3",
            spacing="2",
        ),
        spacing="2",
        align="start",
    )


def _spike_table() -> rx.Component:
    """Render the per-spike metrics table.

    Returns:
        A scrollable table listing metrics for each detected spike.
    """
    return rx.vstack(
        rx.text("Per-spike metrics", size="1", weight="bold"),
        rx.scroll_area(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell(rx.text("#", size="1")),
                        rx.table.column_header_cell(
                            rx.text("Threshold (mV)", size="1")
                        ),
                        rx.table.column_header_cell(rx.text("Peak (mV)", size="1")),
                        rx.table.column_header_cell(
                            rx.text("Rise time (ms)", size="1")
                        ),
                        rx.table.column_header_cell(
                            rx.text("Half-width (ms)", size="1")
                        ),
                        rx.table.column_header_cell(rx.text("AHP (mV)", size="1")),
                    ),
                ),
                rx.table.body(
                    rx.foreach(AppState.ap_metrics, _spike_row),
                ),
                size="1",
                variant="surface",
                width="100%",
            ),
            max_height="120px",
        ),
        spacing="2",
        width="100%",
        align="start",
    )


def metrics_panel() -> rx.Component:
    """Render the AP analysis metrics panel.

    The panel is conditionally displayed only when AP metrics are available
    (i.e. after a single-sweep current clamp simulation).  It shows a summary
    statistics section and a per-spike detail table.

    Returns:
        A Reflex component containing the full metrics panel.
    """
    return rx.cond(
        AppState.has_ap_metrics,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("activity", size=14),
                    rx.text("Action Potential Metrics", size="2", weight="bold"),
                    spacing="2",
                    align="center",
                ),
                rx.hstack(
                    _summary_section(),
                    rx.separator(orientation="vertical"),
                    _spike_table(),
                    spacing="4",
                    align="start",
                    width="100%",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            padding_x="4",
            padding_y="3",
            border_top="1px solid var(--gray-4)",
            width="100%",
        ),
        rx.fragment(),
    )
