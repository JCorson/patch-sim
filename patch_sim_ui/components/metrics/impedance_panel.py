"""Impedance-analysis panel: resonance summary and |Z| / phase plot.

Shown in the analysis sidebar in place of the AP-metrics panel when the
active current-clamp protocol is a chirp (Frequency Response).
"""

import reflex as rx

from patch_sim_ui.state.analysis import AnalysisState


def _impedance_summary() -> rx.Component:
    """Render the resonance-summary section (resonance frequency, Q, peak |Z|).

    Returns:
        A compact grid of labeled impedance metrics drawn from
        ``AnalysisState.impedance_data``.
    """
    d = AnalysisState.impedance_data
    return rx.box(
        rx.grid(
            rx.text("Resonance f_R", size="1", color="gray"),
            rx.text(d["resonance_frequency"].to(str) + " Hz", size="1", align="left"),
            rx.text("Quality Q", size="1", color="gray"),
            rx.text(d["quality_factor"].to(str), size="1", align="left"),
            rx.text("Peak |Z|", size="1", color="gray"),
            rx.text(
                d["peak_impedance"].to(str) + " " + d["units"].to(str),
                size="1",
                align="left",
            ),
            columns="2",
            spacing="2",
            width="100%",
        ),
        padding="3",
        border_bottom="1px solid var(--gray-4)",
        width="100%",
    )


def _impedance_plot() -> rx.Component:
    """Render the embedded impedance-profile plot (|Z| and phase vs frequency).

    Includes a sub-window caption when ``analyze_impedance`` recovered the
    profile from a spike-free segment rather than the full chirp window.

    Returns:
        A flex container holding the optional caption and the impedance Plotly
        figure.
    """
    return rx.flex(
        rx.cond(
            AnalysisState.impedance_caption != "",
            rx.text(
                AnalysisState.impedance_caption,
                size="1",
                color="gray",
                text_align="center",
                padding_x="3",
                padding_top="2",
            ),
            rx.fragment(),
        ),
        rx.plotly(
            data=AnalysisState.impedance_figure,
            width="100%",
        ),
        direction="column",
        width="100%",
        flex_shrink="0",
        border_top="1px solid var(--gray-4)",
        padding="1",
    )


def impedance_tab() -> rx.Component:
    """Render the impedance-analysis tab for the chirp current-clamp pane.

    Shows the resonance summary and the |Z| / phase plot when impedance data
    is available, the specific unavailability reason when the analysis bailed
    out, and a generic placeholder otherwise.

    Returns:
        The impedance-analysis content as a scrollable flex column.
    """
    return rx.cond(
        AnalysisState.has_impedance_data,
        rx.scroll_area(
            rx.flex(
                _impedance_summary(),
                _impedance_plot(),
                direction="column",
                width="100%",
            ),
            height="100%",
            width="100%",
        ),
        rx.cond(
            AnalysisState.impedance_unavailable_reason != "",
            rx.flex(
                rx.text(
                    "Impedance not computed: "
                    + AnalysisState.impedance_unavailable_reason,
                    size="1",
                    color="gray",
                    text_align="center",
                ),
                padding="4",
                justify="center",
            ),
            rx.flex(
                rx.text(
                    "Run a Frequency Response (chirp) current clamp simulation "
                    "to see the impedance profile.  The chirp must keep the "
                    "cell subthreshold — reduce the amplitude and, for cells "
                    "that fire spontaneously, apply a hyperpolarizing DC "
                    "offset (holding current).",
                    size="1",
                    color="gray",
                    text_align="center",
                ),
                padding="4",
                justify="center",
            ),
        ),
    )
