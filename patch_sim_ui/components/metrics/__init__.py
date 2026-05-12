"""Right-hand analysis sidebar — split into per-analysis panels.

A collapsible panel that automatically shows the relevant analysis view:
the impedance profile for a chirp current-clamp protocol, AP metrics for any
other current-clamp protocol, and the I-V curve in voltage clamp mode.

The shell pieces (header, passive-membrane row, expand/collapse strips,
``analysis_sidebar`` entry point) live in this ``__init__.py``; the
per-analysis content lives in sibling modules:

- :mod:`patch_sim_ui.components.metrics.ap_panel` — AP metrics tab composer
- :mod:`patch_sim_ui.components.metrics.burst_panel` — burst summary/table
- :mod:`patch_sim_ui.components.metrics.calcium_panel` — calcium summary/table
- :mod:`patch_sim_ui.components.metrics.fi_gv_panel` — F-I, g-V, τ-V, I-V tab
- :mod:`patch_sim_ui.components.metrics.impedance_panel` — chirp impedance profile
- :mod:`patch_sim_ui.components.metrics.sfa_hyperpol_panel` — SFA/sag plots
- :mod:`patch_sim_ui.components.metrics._row` — shared dict-to-row helper

Selected private helpers are re-exported so existing tests keep working.
"""

import reflex as rx

from patch_sim.constants import CHIRP_PROTOCOL, CURRENT_CLAMP
from patch_sim_ui.components.metrics.ap_panel import _ap_metrics_tab, _spike_row
from patch_sim_ui.components.metrics.burst_panel import _burst_row
from patch_sim_ui.components.metrics.calcium_panel import _ca_transient_row
from patch_sim_ui.components.metrics.fi_gv_panel import (
    _gv_plot,
    _iv_curve_tab,
    _tau_v_plot,
)
from patch_sim_ui.components.metrics.impedance_panel import impedance_tab
from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.protocol import ProtocolState

_PANEL_WIDTH = "300px"
_COLLAPSED_WIDTH = "36px"


def _passive_metric(label: str, value: rx.Var) -> rx.Component:
    """Render one passive-property metric as a label stacked above its value.

    Args:
        label: Metric name shown above the value (e.g. ``"Input resistance"``).
        value: Reactive string var holding the formatted value with units.

    Returns:
        A small vstack with the gray label on top and the value below.
    """
    return rx.vstack(
        rx.text(label, size="1", color="gray"),
        rx.text(value, size="1"),
        spacing="0",
        align="start",
        min_width="0",
    )


def _membrane_test_section() -> rx.Component:
    """Render the always-visible passive membrane properties section.

    Displays R_in, τₘ, and Cₘ from the dedicated membrane test as three
    columns, each with the value below its label.  The displayed units
    depend on whether the active preset declares a cell surface area:
    absolute MΩ / pF when present, per-area kΩ·cm² / µF/cm² otherwise.
    τₘ is always in ms and is invariant to the conversion.

    A tooltip explains the active mode.

    Returns:
        A three-column grid of labeled passive property values, or an empty
        box when no membrane test results are available.
    """
    s = AnalysisState
    tooltip_text = rx.cond(
        s.mt_units_mode == "absolute",
        (
            "R_n in megaohms; C in picofarads; τ_m = R_n × C. Computed from "
            "the active preset's cell surface area."
        ),
        (
            "Per-area passive properties (kΩ·cm², µF/cm²). The active preset "
            "has no surface area declared, so absolute R_n (MΩ) and C (pF) "
            "are not available."
        ),
    )
    return rx.cond(
        AnalysisState.has_membrane_test,
        rx.tooltip(
            rx.grid(
                _passive_metric(
                    "Input resistance", s.mt_input_resistance + " " + s.mt_r_units
                ),
                _passive_metric("Time constant", s.mt_time_constant + " ms"),
                _passive_metric(
                    "Capacitance", s.mt_membrane_capacitance + " " + s.mt_c_units
                ),
                columns="3",
                spacing="2",
                padding_x="3",
                padding_y="2",
                border_bottom="1px solid var(--gray-4)",
                width="100%",
            ),
            content=tooltip_text,
        ),
        rx.box(),
    )


def _expanded_panel() -> rx.Component:
    """Render the full expanded analysis sidebar.

    The analysis view shown depends on the active clamp mode and protocol:
    the impedance profile for a chirp current-clamp protocol, AP metrics for
    any other current-clamp protocol, and the I-V curve in voltage clamp mode.

    Returns:
        A fixed-width flex column with a header and the mode-appropriate
        analysis content.
    """
    return rx.flex(
        rx.hstack(
            rx.icon("chart-line", size=14),
            rx.cond(
                ProtocolState.clamp_mode == CURRENT_CLAMP,
                rx.cond(
                    ProtocolState.protocol_type == CHIRP_PROTOCOL,
                    rx.text("Impedance Analysis", size="4", weight="bold"),
                    rx.text("AP Analysis", size="4", weight="bold"),
                ),
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
        _membrane_test_section(),
        rx.box(
            rx.cond(
                ProtocolState.clamp_mode == CURRENT_CLAMP,
                rx.cond(
                    ProtocolState.protocol_type == CHIRP_PROTOCOL,
                    impedance_tab(),
                    _ap_metrics_tab(),
                ),
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


__all__ = [
    "analysis_sidebar",
    "_burst_row",
    "_ca_transient_row",
    "_gv_plot",
    "_iv_curve_tab",
    "_spike_row",
    "_tau_v_plot",
]
