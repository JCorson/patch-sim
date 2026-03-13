"""Sweep overlay controls and trace visibility checkboxes."""

import reflex as rx

from ap_sim_ui.state import AppState


def _trace_checkbox(label: str, var: rx.Var, handler) -> rx.Component:
    """Render a single trace visibility checkbox."""
    return rx.hstack(
        rx.checkbox(
            checked=var,
            on_change=handler,
        ),
        rx.text(label, size="2"),
        spacing="2",
        align="center",
    )


def _cc_trace_checkboxes() -> rx.Component:
    """Trace visibility checkboxes for current clamp mode."""
    return rx.vstack(
        _trace_checkbox("Voltage", AppState.show_voltage, AppState.set_show_voltage),
        _trace_checkbox(
            "K activation (n)",
            AppState.show_potassium_activation,
            AppState.set_show_potassium_activation,
        ),
        _trace_checkbox(
            "Na activation (m)",
            AppState.show_sodium_activation,
            AppState.set_show_sodium_activation,
        ),
        _trace_checkbox(
            "Na inactivation (h)",
            AppState.show_sodium_inactivation,
            AppState.set_show_sodium_inactivation,
        ),
        spacing="1",
        wrap="wrap",
        direction="row",
    )


def _vc_trace_checkboxes() -> rx.Component:
    """Trace visibility checkboxes for voltage clamp mode."""
    return rx.vstack(
        _trace_checkbox(
            "Total current",
            AppState.show_total_current,
            AppState.set_show_total_current,
        ),
        _trace_checkbox(
            "I_Na",
            AppState.show_sodium_current,
            AppState.set_show_sodium_current,
        ),
        _trace_checkbox(
            "I_K",
            AppState.show_potassium_current,
            AppState.set_show_potassium_current,
        ),
        _trace_checkbox(
            "I_L (leak)",
            AppState.show_leak_current,
            AppState.set_show_leak_current,
        ),
        _trace_checkbox(
            "K activation (n)",
            AppState.show_potassium_activation,
            AppState.set_show_potassium_activation,
        ),
        _trace_checkbox(
            "Na activation (m)",
            AppState.show_sodium_activation,
            AppState.set_show_sodium_activation,
        ),
        _trace_checkbox(
            "Na inactivation (h)",
            AppState.show_sodium_inactivation,
            AppState.set_show_sodium_inactivation,
        ),
        spacing="1",
        wrap="wrap",
        direction="row",
    )


def _sweep_chip(sweep) -> rx.Component:
    """Render a colour-coded badge for a saved sweep."""
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
    )


def sweep_manager() -> rx.Component:
    """Trace selector checkboxes and sweep overlay management bar."""
    return rx.vstack(
        rx.hstack(
            rx.text("Traces:", size="2", weight="bold"),
            rx.cond(
                AppState.clamp_mode == "Current Clamp",
                _cc_trace_checkboxes(),
                _vc_trace_checkboxes(),
            ),
            spacing="3",
            align="start",
            wrap="wrap",
        ),
        rx.separator(),
        rx.hstack(
            rx.text("Sweeps:", size="2", weight="bold"),
            rx.foreach(AppState.saved_sweeps, _sweep_chip),
            rx.spacer(),
            rx.button(
                "Add sweep",
                on_click=AppState.add_sweep,
                size="1",
                variant="soft",
                disabled=~AppState.has_result,
            ),
            rx.button(
                "Clear",
                on_click=AppState.clear_sweeps,
                size="1",
                variant="soft",
                color_scheme="red",
                disabled=AppState.saved_sweeps.length() == 0,
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
