"""Neuron parameter panel component."""

import reflex as rx

from ap_sim_ui.state import AppState


def _reversal_row(label: str, value: rx.Var, unit: str = "mV") -> rx.Component:
    """Render a read-only reversal potential display row."""
    return rx.hstack(
        rx.text(label, size="2", color="gray"),
        rx.spacer(),
        rx.badge(
            rx.text(value.to_string(), size="2"),
            rx.text(f" {unit}", size="1", color="gray"),
            variant="soft",
        ),
        width="100%",
    )


def neuron_panel() -> rx.Component:
    """Sidebar panel for configuring Hodgkin-Huxley neuron parameters."""
    return rx.vstack(
        rx.heading("Neuron Parameters", size="3"),
        rx.separator(),
        rx.text("Conductances (mS/cm²)", size="2", weight="bold"),
        rx.vstack(
            rx.hstack(
                rx.text("g_Na", size="2", color="gray"),
                rx.spacer(),
                rx.input(
                    value=AppState.g_Na,
                    on_change=AppState.set_g_Na,
                    width="90px",
                    size="1",
                    type="number",
                    min="0",
                    max="300",
                    step="1",
                ),
                width="100%",
            ),
            rx.slider(
                min=0,
                max=300,
                step=1,
                value=[AppState.g_Na],
                on_change=AppState.set_g_Na,
                width="100%",
            ),
            rx.hstack(
                rx.text("g_K", size="2", color="gray"),
                rx.spacer(),
                rx.input(
                    value=AppState.g_K,
                    on_change=AppState.set_g_K,
                    width="90px",
                    size="1",
                    type="number",
                    min="0",
                    max="100",
                    step="0.5",
                ),
                width="100%",
            ),
            rx.slider(
                min=0,
                max=100,
                step=0.5,
                value=[AppState.g_K],
                on_change=AppState.set_g_K,
                width="100%",
            ),
            rx.hstack(
                rx.text("g_L", size="2", color="gray"),
                rx.spacer(),
                rx.input(
                    value=AppState.g_L,
                    on_change=AppState.set_g_L,
                    width="90px",
                    size="1",
                    type="number",
                    min="0",
                    max="2",
                    step="0.01",
                ),
                width="100%",
            ),
            rx.slider(
                min=0,
                max=2,
                step=0.01,
                value=[AppState.g_L],
                on_change=AppState.set_g_L,
                width="100%",
            ),
            spacing="1",
            width="100%",
        ),
        rx.separator(),
        rx.text("Membrane Properties", size="2", weight="bold"),
        rx.hstack(
            rx.text("C_m (µF/cm²)", size="2", color="gray"),
            rx.spacer(),
            rx.input(
                value=AppState.C_m,
                on_change=AppState.set_C_m,
                width="90px",
                size="1",
                type="number",
                min="0.1",
                max="5",
                step="0.1",
            ),
            width="100%",
        ),
        rx.slider(
            min=0.1,
            max=5,
            step=0.1,
            value=[AppState.C_m],
            on_change=AppState.set_C_m,
            width="100%",
        ),
        rx.hstack(
            rx.text("v_rest (mV)", size="2", color="gray"),
            rx.spacer(),
            rx.input(
                value=AppState.v_rest,
                on_change=AppState.set_v_rest,
                width="90px",
                size="1",
                type="number",
                min="-90",
                max="-40",
                step="1",
            ),
            width="100%",
        ),
        rx.slider(
            min=-90,
            max=-40,
            step=1,
            value=[AppState.v_rest],
            on_change=AppState.set_v_rest,
            width="100%",
        ),
        rx.hstack(
            rx.text("T (K)", size="2", color="gray"),
            rx.spacer(),
            rx.input(
                value=AppState.T,
                on_change=AppState.set_T,
                width="90px",
                size="1",
                type="number",
                min="273.15",
                max="323.15",
                step="0.5",
            ),
            width="100%",
        ),
        rx.slider(
            min=273.15,
            max=323.15,
            step=0.5,
            value=[AppState.T],
            on_change=AppState.set_T,
            width="100%",
        ),
        rx.separator(),
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text(
                    "Ion Concentrations (mM)",
                    size="2",
                    weight="bold",
                    color="var(--gray-12)",
                ),
                content=rx.vstack(
                    _ion_row("Na⁺ out", AppState.Na_out, AppState.set_Na_out, 1, 500),
                    _ion_row("Na⁺ in", AppState.Na_in, AppState.set_Na_in, 1, 100),
                    _ion_row("K⁺ out", AppState.K_out, AppState.set_K_out, 1, 50),
                    _ion_row("K⁺ in", AppState.K_in, AppState.set_K_in, 1, 300),
                    _ion_row("Cl⁻ out", AppState.Cl_out, AppState.set_Cl_out, 1, 300),
                    _ion_row("Cl⁻ in", AppState.Cl_in, AppState.set_Cl_in, 1, 100),
                    spacing="2",
                    width="100%",
                ),
                value="ion-concentrations",
            ),
            collapsible=True,
            variant="ghost",
            width="100%",
        ),
        rx.separator(),
        rx.text("Reversal Potentials", size="2", weight="bold"),
        rx.vstack(
            rx.hstack(
                rx.text("E_Na", size="2", color="gray"),
                rx.spacer(),
                rx.badge(
                    rx.text(AppState.E_Na.to(str), size="2"),
                    rx.text(" mV", size="1", color="gray"),
                    variant="soft",
                ),
                width="100%",
            ),
            rx.hstack(
                rx.text("E_K", size="2", color="gray"),
                rx.spacer(),
                rx.badge(
                    rx.text(AppState.E_K.to(str), size="2"),
                    rx.text(" mV", size="1", color="gray"),
                    variant="soft",
                ),
                width="100%",
            ),
            rx.hstack(
                rx.text("E_L", size="2", color="gray"),
                rx.spacer(),
                rx.badge(
                    rx.text(AppState.E_L.to(str), size="2"),
                    rx.text(" mV", size="1", color="gray"),
                    variant="soft",
                ),
                width="100%",
            ),
            spacing="1",
            width="100%",
        ),
        spacing="3",
        width="100%",
        padding="4",
    )


def _ion_row(
    label: str,
    var: rx.Var,
    handler,
    min_val: float,
    max_val: float,
) -> rx.Component:
    """Render a single ion concentration row with input."""
    return rx.hstack(
        rx.text(label, size="2", color="gray", width="70px"),
        rx.slider(
            min=min_val,
            max=max_val,
            step=1,
            value=[var],
            on_change=handler,
            width="100%",
        ),
        rx.input(
            value=var,
            on_change=handler,
            width="80px",
            size="1",
            type="number",
            min=str(min_val),
            max=str(max_val),
        ),
        width="100%",
        spacing="2",
    )
