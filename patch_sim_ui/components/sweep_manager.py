"""Sweep overlay controls and trace visibility popover."""

import reflex as rx

from patch_sim_ui.channels import ADDITIONAL_CHANNELS
from patch_sim_ui.constants import CURRENT_CLAMP
from patch_sim_ui.state import SimulationState
from patch_sim_ui.state.log import LogState
from patch_sim_ui.state.neuron import NeuronState
from patch_sim_ui.state.protocol import ProtocolState
from patch_sim_ui.state.visibility import VisibilityState


def _trace_checkbox(label: str, var: rx.Var, handler) -> rx.Component:
    """Render a single trace visibility checkbox.

    Args:
        label: Display text next to the checkbox.
        var: State var bound to the checkbox checked state.
        handler: Event handler called on change.

    Returns:
        An hstack containing a checkbox and label.
    """
    return rx.hstack(
        rx.checkbox(
            checked=var,
            on_change=handler,
        ),
        rx.text(label, size="1"),
        spacing="2",
        align="center",
    )


def _section_label(text: str) -> rx.Component:
    """Render a bold gray section header inside the popover.

    Args:
        text: Header text to display.

    Returns:
        A styled text component used as a section label.
    """
    return rx.text(text, size="1", weight="bold", color="gray")


def _channel_trace_group(
    header: str,
    enabled_var: rx.Var,
    current_label: str,
    current_var: rx.Var,
    current_handler,
    gating_label: str,
    gating_var: rx.Var,
    gating_handler,
    *,
    include_current: bool = True,
) -> rx.Component:
    """Render a per-channel group with optional current and required gating checkboxes.

    The group is wrapped in ``rx.cond`` and only rendered when the channel is
    enabled.

    Args:
        header: Section header text (e.g. ``"Ih (HCN)"``).
        enabled_var: State var indicating whether the channel is enabled.
        current_label: Label for the current trace checkbox.
        current_var: State var for current trace visibility.
        current_handler: Event handler for current trace checkbox.
        gating_label: Label for the gating trace checkbox.
        gating_var: State var for gating trace visibility.
        gating_handler: Event handler for gating trace checkbox.
        include_current: When False, omit the current checkbox. Use False for
            current clamp mode where additional channel currents are not plotted.

    Returns:
        A fragment containing the conditional channel group.
    """
    current_row = (
        [_trace_checkbox(current_label, current_var, current_handler)]
        if include_current
        else []
    )
    return rx.cond(
        enabled_var,
        rx.fragment(
            rx.separator(),
            _section_label(header),
            *current_row,
            _trace_checkbox(gating_label, gating_var, gating_handler),
        ),
    )


# (header, enabled_var, current_label, current_var, current_handler,
#  gating_label, gating_var, gating_handler)
# Derived from the channel registry in patch_sim_ui.channels.
_ADDITIONAL_CHANNEL_TRACE_SPECS = [
    (
        ch.label,
        getattr(NeuronState, ch.enabled_field),
        ch.current_label,
        getattr(VisibilityState, ch.current_visibility_field),
        getattr(VisibilityState, f"set_{ch.current_visibility_field}"),
        ch.gating_label,
        getattr(VisibilityState, ch.gating_visibility_field),
        getattr(VisibilityState, f"set_{ch.gating_visibility_field}"),
    )
    for ch in ADDITIONAL_CHANNELS
]


def _additional_channels_section(*, include_current: bool = True) -> rx.Component:
    """Render conditional trace groups for all additional channels.

    Each group is only shown when the corresponding channel is enabled.

    Args:
        include_current: When False, omit the current checkbox from each group.
            Pass False for current clamp mode where additional channel currents
            are not plotted.

    Returns:
        A fragment containing all additional channel trace groups.
    """
    return rx.fragment(
        *[
            _channel_trace_group(*spec, include_current=include_current)
            for spec in _ADDITIONAL_CHANNEL_TRACE_SPECS
        ]
    )


def _cc_popover_content() -> rx.Component:
    """Popover content for current clamp trace visibility.

    Organises checkboxes into Response, Gating Variables, and Additional
    Channels sections.

    Returns:
        A scrollable vstack of labelled checkbox groups.
    """
    return rx.vstack(
        _section_label("Response"),
        _trace_checkbox(
            "Voltage",
            VisibilityState.show_voltage,
            VisibilityState.set_show_voltage,
        ),
        rx.separator(),
        _section_label("Gating Variables"),
        _trace_checkbox(
            "K activation (n)",
            VisibilityState.show_potassium_activation,
            VisibilityState.set_show_potassium_activation,
        ),
        _trace_checkbox(
            "Na activation (m)",
            VisibilityState.show_sodium_activation,
            VisibilityState.set_show_sodium_activation,
        ),
        _trace_checkbox(
            "Na inactivation (h)",
            VisibilityState.show_sodium_inactivation,
            VisibilityState.set_show_sodium_inactivation,
        ),
        _additional_channels_section(include_current=False),
        spacing="1",
        padding="2",
        max_height="60vh",
        overflow_y="auto",
    )


def _vc_popover_content() -> rx.Component:
    """Popover content for voltage clamp trace visibility.

    Organises checkboxes into Currents, Gating Variables, and Additional
    Channels sections.

    Returns:
        A scrollable vstack of labelled checkbox groups.
    """
    return rx.vstack(
        _section_label("Currents"),
        _trace_checkbox(
            "Total current",
            VisibilityState.show_total_current,
            VisibilityState.set_show_total_current,
        ),
        _trace_checkbox(
            "I_Na",
            VisibilityState.show_sodium_current,
            VisibilityState.set_show_sodium_current,
        ),
        _trace_checkbox(
            "I_K",
            VisibilityState.show_potassium_current,
            VisibilityState.set_show_potassium_current,
        ),
        _trace_checkbox(
            "I_NaL (Na⁺ leak)",
            VisibilityState.show_na_leak_current,
            VisibilityState.set_show_na_leak_current,
        ),
        _trace_checkbox(
            "I_KL (K⁺ leak)",
            VisibilityState.show_k_leak_current,
            VisibilityState.set_show_k_leak_current,
        ),
        rx.separator(),
        _section_label("Gating Variables"),
        _trace_checkbox(
            "K activation (n)",
            VisibilityState.show_potassium_activation,
            VisibilityState.set_show_potassium_activation,
        ),
        _trace_checkbox(
            "Na activation (m)",
            VisibilityState.show_sodium_activation,
            VisibilityState.set_show_sodium_activation,
        ),
        _trace_checkbox(
            "Na inactivation (h)",
            VisibilityState.show_sodium_inactivation,
            VisibilityState.set_show_sodium_inactivation,
        ),
        _additional_channels_section(),
        spacing="1",
        padding="2",
        max_height="60vh",
        overflow_y="auto",
    )


def _trace_visibility_popover() -> rx.Component:
    """Render the Traces popover button with mode-conditional content.

    Returns:
        A popover root containing a trigger button and the appropriate
        popover content for the current clamp mode.
    """
    return rx.popover.root(
        rx.popover.trigger(
            rx.button(
                "\U0001f441 Traces",
                size="1",
                variant="soft",
            ),
        ),
        rx.popover.content(
            rx.cond(
                ProtocolState.clamp_mode == CURRENT_CLAMP,
                _cc_popover_content(),
                _vc_popover_content(),
            ),
        ),
    )


def _chip(sweep, color_scheme: str | None = None) -> rx.Component:
    """Render a colour-coded badge for a sweep or stored trace.

    Args:
        sweep: A ``Sweep`` instance.
        color_scheme: Optional Radix color scheme (e.g. ``"orange"`` for stored
            traces).

    Returns:
        A badge component showing the sweep colour dot and label.
    """
    extra = {"color_scheme": color_scheme} if color_scheme is not None else {}
    return rx.badge(
        rx.box(
            width="10px",
            height="10px",
            border_radius="50%",
            background_color=sweep.color,
            display="inline-block",
            margin_right="4px",
        ),
        sweep.label,
        variant="outline",
        **extra,
    )


def _stored_chip(sweep) -> rx.Component:
    """Render a badge for a stored trace (for use with rx.foreach)."""
    return _chip(sweep)


def sweep_manager() -> rx.Component:
    """Trace visibility popover and stored-trace management bar.

    Returns:
        A vstack containing the Traces popover button, hover toggle, and the
        stored-trace controls (Store, Clear Stored).
    """
    return rx.vstack(
        rx.hstack(
            _trace_visibility_popover(),
            rx.tooltip(
                rx.icon_button(
                    rx.icon("crosshair"),
                    on_click=SimulationState.toggle_hover,
                    size="1",
                    variant=rx.cond(SimulationState.show_hover, "soft", "outline"),
                    color_scheme=rx.cond(SimulationState.show_hover, "gray", "gray"),
                    aria_label="Toggle hover tooltips",
                ),
                content=rx.cond(
                    SimulationState.show_hover,
                    "Hide hover tooltips",
                    "Show hover tooltips",
                ),
            ),
            rx.separator(orientation="vertical"),
            rx.button(
                rx.icon("scroll-text"),
                "Logs",
                on_click=LogState.toggle_log_panel,
                size="1",
                variant="soft",
            ),
            rx.separator(orientation="vertical"),
            rx.text("Stored:", size="2", weight="bold"),
            rx.foreach(SimulationState.stored_traces, _stored_chip),
            rx.tooltip(
                rx.button(
                    "Store",
                    on_click=SimulationState.store_trace,
                    size="1",
                    variant="soft",
                    color_scheme="blue",
                    disabled=(
                        ~SimulationState.has_result | SimulationState.is_multi_sweep
                    ),
                ),
                content=rx.cond(
                    SimulationState.is_multi_sweep,
                    "Store is unavailable for multi-sweep protocols",
                    "Store current trace as a reference overlay",
                ),
            ),
            rx.button(
                "Clear Stored",
                on_click=SimulationState.clear_stored_traces,
                size="1",
                variant="soft",
                color_scheme="red",
                disabled=~SimulationState.has_stored_traces,
            ),
            spacing="2",
            align="center",
            wrap="wrap",
            width="100%",
        ),
        width="100%",
        padding_x="4",
        padding_y="2",
    )
