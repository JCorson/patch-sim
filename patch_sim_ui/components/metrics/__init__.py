"""Right-hand analysis sidebar — split into per-analysis panels.

A collapsible panel that automatically shows the relevant analysis view
based on the active clamp mode: AP metrics for current clamp, I-V curve
for voltage clamp.

The shell pieces (header, passive-membrane row, expand/collapse strips,
``analysis_sidebar`` entry point) live in this ``__init__.py``; the
per-analysis content lives in sibling modules:

- :mod:`patch_sim_ui.components.metrics.ap_panel` — AP metrics tab composer
- :mod:`patch_sim_ui.components.metrics.burst_panel` — burst summary/table
- :mod:`patch_sim_ui.components.metrics.calcium_panel` — calcium summary/table
- :mod:`patch_sim_ui.components.metrics.fi_gv_panel` — F-I, g-V, τ-V, I-V tab
- :mod:`patch_sim_ui.components.metrics.sfa_hyperpol_panel` — SFA/sag plots
- :mod:`patch_sim_ui.components.metrics._row` — shared dict-to-row helper

Selected private helpers are re-exported so existing tests keep working.
"""

import reflex as rx

from patch_sim.constants import CURRENT_CLAMP
from patch_sim_ui.components.metrics.ap_panel import _ap_metrics_tab, _spike_row
from patch_sim_ui.components.metrics.burst_panel import _burst_row
from patch_sim_ui.components.metrics.calcium_panel import _ca_transient_row
from patch_sim_ui.components.metrics.fi_gv_panel import (
    _gv_plot,
    _iv_curve_tab,
    _tau_v_plot,
)
from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.protocol import ProtocolState

_PANEL_WIDTH = "300px"
_COLLAPSED_WIDTH = "36px"


def _membrane_test_section() -> rx.Component:
    """Render the always-visible passive membrane properties section.

    Displays R_in, τₘ, and Cₘ from the dedicated membrane test on a single
    compact row.  The displayed units depend on whether the active preset
    declares a cell surface area: absolute MΩ / pF when present, per-area
    kΩ·cm² / µF/cm² otherwise.  τₘ is always in ms and is invariant to the
    conversion.

    A tooltip explains the active mode.

    Returns:
        A single-row hstack of labeled passive property values, or an empty
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
            rx.hstack(
                rx.text("R_in", size="1", color="gray"),
                rx.text(s.mt_input_resistance + " " + s.mt_r_units, size="1"),
                rx.text("τ_m", size="1", color="gray", padding_left="2"),
                rx.text(s.mt_time_constant + " ms", size="1"),
                rx.text("C_m", size="1", color="gray", padding_left="2"),
                rx.text(s.mt_membrane_capacitance + " " + s.mt_c_units, size="1"),
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
        _membrane_test_section(),
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


__all__ = [
    "analysis_sidebar",
    "_burst_row",
    "_ca_transient_row",
    "_gv_plot",
    "_iv_curve_tab",
    "_spike_row",
    "_tau_v_plot",
]
