"""Right-hand analysis sidebar component.

A collapsible panel that automatically shows the relevant analysis view
based on the active clamp mode: AP metrics for current clamp, I-V curve
for voltage clamp.
"""

import reflex as rx

from patch_sim.constants import CURRENT_CLAMP
from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.protocol import ProtocolState

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
    return rx.table.row(
        rx.table.cell(rx.text(transient["index"].to(int) + 1, size="1")),
        rx.table.cell(
            rx.cond(
                transient["sweep_index"].to(int) > 0,
                rx.text(transient["sweep_index"].to(int), size="1"),
                rx.text("", size="1"),
            ),
        ),
        rx.table.cell(rx.text(transient["peak_time"].to(str), size="1")),
        rx.table.cell(rx.text(transient["peak_concentration"].to(str), size="1")),
        rx.table.cell(rx.text(transient["time_to_peak"].to(str), size="1")),
        rx.table.cell(rx.text(transient["decay_tau"].to(str), size="1")),
        rx.table.cell(rx.text(transient["amplitude"].to(str), size="1")),
    )


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


def _hyperpolarization_plot() -> rx.Component:
    """Render the sag/rebound plot inside the AP Metrics tab.

    Shown only when hyperpolarization analysis data is available (current clamp
    multi-sweep with all-negative steps).

    Returns:
        A compact Plotly sag/rebound figure inside a flex container.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.hyperpolarization_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _ap_sfa_plot() -> rx.Component:
    """Render the embedded SFA plot inside the AP Metrics tab.

    Shown whenever SFA data is available — for single-sweep runs this is one
    curve with an adaptation-index annotation; for multi-sweep runs it is one
    curve per sweep.

    Returns:
        A compact Plotly SFA figure inside a flex container.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.sfa_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _ap_phase_plane_plot() -> rx.Component:
    """Render the V vs dV/dt phase-plane plot inside the Analysis tab.

    Shows the membrane voltage on the x-axis against its time derivative on
    the y-axis.  One trajectory is drawn per sweep; multi-sweep runs display
    trajectories in their respective sweep colours.

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

    Shows AP summary statistics, the calcium-transient summary (when calcium
    dynamics were active), the F-I curve (multi-sweep), the SFA curve, and
    the V vs dV/dt phase-plane when data are available.  Displays a
    placeholder when no data exists yet.

    Returns:
        The analysis sub-tab content as a scrollable flex column.
    """
    return rx.cond(
        AnalysisState.has_ap_or_fi | AnalysisState.has_ca_transient_metrics,
        rx.scroll_area(
            rx.flex(
                rx.cond(AnalysisState.has_ap_metrics, _ap_summary(), rx.box()),
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


def _membrane_test_section() -> rx.Component:
    """Render the always-visible passive membrane properties section.

    Displays R_in, τₘ, and Cₘ from the dedicated membrane test on a single
    compact row.  The displayed units depend on whether the active neuron has
    an ``area_cm2`` set: absolute MΩ / pF when present, per-area kΩ·cm² /
    µF/cm² otherwise.  τₘ is always in ms and is invariant to the conversion.

    A tooltip explains the active mode and how to switch.

    Returns:
        A single-row hstack of labelled passive property values, or an empty
        box when no membrane test results are available.
    """
    s = AnalysisState
    tooltip_text = rx.cond(
        s.mt_units_mode == "absolute",
        (
            "R_n in megaohms; C in picofarads; τ_m = R_n × C. Computed from "
            "the cell surface area set in Neuron Parameters → Membrane "
            "Properties → Cell area."
        ),
        (
            "Per-area passive properties (kΩ·cm², µF/cm²). Set the cell "
            "surface area in Neuron Parameters → Membrane Properties to "
            "convert to absolute R_n (MΩ) and C (pF)."
        ),
    )
    return rx.cond(
        AnalysisState.has_membrane_test,
        rx.tooltip(
            rx.hstack(
                rx.text("R_in", size="1", color="gray"),
                rx.text(s.mt_input_resistance + " " + s.mt_r_units, size="1"),
                rx.text("τ_m", size="1", color="gray", padding_left="2"),
                rx.text(s.mt_time_constant + " ms", size="1"),
                rx.text("C_m", size="1", color="gray", padding_left="2"),
                rx.text(
                    s.mt_membrane_capacitance + " " + s.mt_c_units, size="1"
                ),
                padding_x="3",
                padding_y="2",
                border_bottom="1px solid var(--gray-4)",
                width="100%",
                align="center",
                wrap="nowrap",
            ),
            content=tooltip_text,
        ),
        rx.box(),
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


def _ap_metrics_tab() -> rx.Component:
    """Render the full AP Metrics panel with three sub-tabs.

    Sub-tabs:
    - **Analysis**: AP summary statistics, calcium-transient summary, F-I
      curve, SFA curve, phase plane.
    - **Spikes**: Per-spike detail table.
    - **Calcium**: Per-transient detail table (only relevant when calcium
      dynamics are active).

    Returns:
        A tabbed flex column for the CC analysis pane.
    """
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Analysis", value="analysis", size="1"),
            rx.tabs.trigger("Spikes", value="spikes", size="1"),
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
    below the I-V curve.  When calcium dynamics were active, the calcium
    transient summary is shown above the I-V curve.  Displays a placeholder
    message when no data exists.

    Returns:
        The full tab content as a flex column.
    """
    return rx.cond(
        AnalysisState.has_iv_data | AnalysisState.has_ca_transient_metrics,
        rx.scroll_area(
            rx.flex(
                rx.cond(
                    AnalysisState.has_ca_transient_metrics,
                    _ca_transient_summary(),
                    rx.box(),
                ),
                rx.cond(
                    AnalysisState.has_iv_data,
                    rx.plotly(
                        data=AnalysisState.iv_figure,
                        width="100%",
                    ),
                    rx.box(),
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

    The analysis view shown depends on the active clamp mode: AP metrics
    are shown in current clamp mode, and the I-V curve is shown in voltage
    clamp mode.

    Returns:
        A fixed-width flex column with a header and the mode-appropriate
        analysis content.
    """
    return rx.flex(
        # Header
        rx.hstack(
            rx.icon("chart-line", size=14),
            rx.cond(
                ProtocolState.clamp_mode == CURRENT_CLAMP,
                rx.text("AP Analysis", size="4", weight="bold"),
                rx.text("I-V Analysis", size="4", weight="bold"),
            ),
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
        # Passive membrane properties — always visible in both clamp modes
        _membrane_test_section(),
        # Analysis content — switches automatically with clamp mode
        rx.box(
            rx.cond(
                ProtocolState.clamp_mode == CURRENT_CLAMP,
                _ap_metrics_tab(),
                _iv_curve_tab(),
            ),
            display="flex",
            flex_direction="column",
            flex_grow="1",
            min_height="0",
            overflow="hidden",
            width="100%",
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

    Toggles between a full panel (with a mode-appropriate analysis view) and
    a thin collapsed strip.  The panel is toggled via
    SimulationState.toggle_analysis_panel.

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
