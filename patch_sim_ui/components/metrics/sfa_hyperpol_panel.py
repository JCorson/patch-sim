"""SFA and hyperpolarization plot panels.

These are small embedded Plotly figures shown inside the AP analysis tab
when the corresponding data is available.
"""

import reflex as rx

from patch_sim_ui.state.analysis import AnalysisState


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
