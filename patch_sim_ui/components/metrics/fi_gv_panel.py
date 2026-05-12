"""F-I, g-V, h∞, τ-V, and I-V curve panels.

Embedded Plotly figures for current-clamp F-I curves and voltage-clamp I-V
analysis (with optional g-V activation Boltzmann fit, h∞ steady-state
inactivation Boltzmann fit, and τ-V activation/inactivation time constants).
The :func:`_iv_curve_tab` composer also reuses the calcium-transient summary
when calcium dynamics were active.
"""

import reflex as rx

from patch_sim_ui.components.metrics.calcium_panel import _ca_transient_summary
from patch_sim_ui.state.analysis import AnalysisState


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


def _inactivation_plot() -> rx.Component:
    """Render the h∞ steady-state inactivation curve plot.

    Shown only for the two-pulse Inactivation voltage-clamp protocol.

    Returns:
        A bordered flex container with the h∞ Plotly figure embedded below
        the I-V curve.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.inactivation_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def _tau_v_plot() -> rx.Component:
    """Render the τ-V (activation/inactivation time constants) plot.

    Returns:
        A bordered flex container with the τ-V Plotly figure embedded
        below the g-V curve.
    """
    return rx.flex(
        rx.plotly(
            data=AnalysisState.tau_v_figure,
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
    also available, a normalized conductance plot with Boltzmann fit is shown
    below the I-V curve; when h∞ steady-state inactivation data is available
    (the two-pulse Inactivation protocol), a normalized availability plot with
    a decreasing Boltzmann fit is shown — in that case the I-V curve above it
    is the same data before normalization.  When calcium dynamics were active,
    the calcium transient summary is shown above the I-V curve.  Displays a
    placeholder message when no data exists.

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
                rx.cond(
                    AnalysisState.has_inactivation_data,
                    _inactivation_plot(),
                    rx.box(),
                ),
                rx.cond(
                    AnalysisState.has_tau_v_data,
                    _tau_v_plot(),
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
