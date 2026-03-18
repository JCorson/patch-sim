"""Neuron parameter panel component."""

import reflex as rx

from ap_sim_ui.constants import PARAM_RANGES
from ap_sim_ui.state import AppState


def _additional_channel_row(
    label: str,
    enabled_var: rx.Var,
    enabled_setter,
    g_var: rx.Var,
    g_setter,
    param_key: str,
) -> rx.Component:
    """Render a checkbox + conditional conductance slider for an additional channel.

    Args:
        label: Display name for the channel (e.g. 'Ih (HCN)').
        enabled_var: Reactive bool var bound to the enable checkbox.
        enabled_setter: Event handler called when the checkbox changes.
        g_var: Reactive float var bound to the conductance slider.
        g_setter: Event handler called when the conductance changes.
        param_key: Key into PARAM_RANGES for the conductance slider bounds.

    Returns:
        A vstack containing an enable checkbox and a conditional g_max row.
    """
    min_val, max_val, step = PARAM_RANGES[param_key]
    return rx.vstack(
        rx.hstack(
            rx.checkbox(
                checked=enabled_var,
                on_change=enabled_setter,
            ),
            rx.text(label, size="2"),
            spacing="2",
            align="center",
        ),
        rx.cond(
            enabled_var,
            _param_row(
                "g_max (mS/cm²)",
                g_var,
                g_setter,
                min_val,
                max_val,
                step,
            ),
        ),
        spacing="2",
        width="100%",
    )


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


def _param_row(
    label: str,
    var: rx.Var,
    handler,
    min_val: float,
    max_val: float,
    step: float,
) -> rx.Component:
    """Render a parameter label, number input, and slider as a grouped block.

    Args:
        label: Display label shown to the left of the input.
        var: Reactive state variable bound to the input and slider.
        handler: Event handler called on input change and slider change.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        step: Increment step for the input and slider.

    Returns:
        A vstack containing an hstack (label + input) and a slider.
    """
    return rx.vstack(
        rx.hstack(
            rx.text(label, size="2", color="gray"),
            rx.spacer(),
            rx.input(
                value=var,
                on_change=handler,
                width="90px",
                size="1",
                type="number",
                min=str(min_val),
                max=str(max_val),
                step=str(step),
            ),
            width="100%",
        ),
        rx.slider(
            min=min_val,
            max=max_val,
            step=step,
            value=[var],
            on_change=handler,
            width="100%",
        ),
        spacing="1",
        width="100%",
    )


def neuron_panel() -> rx.Component:
    """Sidebar panel for configuring Hodgkin-Huxley neuron parameters."""
    return rx.vstack(
        rx.heading("Neuron Parameters", size="3"),
        rx.separator(),
        rx.text("Conductances (mS/cm²)", size="2", weight="bold"),
        rx.vstack(
            _param_row("g_Na", AppState.g_Na, AppState.set_g_Na, *PARAM_RANGES["g_Na"]),
            _param_row("g_K", AppState.g_K, AppState.set_g_K, *PARAM_RANGES["g_K"]),
            _param_row("g_L", AppState.g_L, AppState.set_g_L, *PARAM_RANGES["g_L"]),
            spacing="1",
            width="100%",
        ),
        rx.separator(),
        rx.text("Membrane Properties", size="2", weight="bold"),
        _param_row(
            "C_m (µF/cm²)", AppState.C_m, AppState.set_C_m, *PARAM_RANGES["C_m"]
        ),
        _param_row(
            "v_rest (mV)", AppState.v_rest, AppState.set_v_rest, *PARAM_RANGES["v_rest"]
        ),
        _param_row("T (K)", AppState.T, AppState.set_T, *PARAM_RANGES["T"]),
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
                    _ion_row(
                        "Ca²⁺ out",
                        AppState.Ca_out,
                        AppState.set_Ca_out,
                        0.1,
                        20,
                        0.1,
                    ),
                    _ion_row(
                        "Ca²⁺ in",
                        AppState.Ca_in,
                        AppState.set_Ca_in,
                        0.00001,
                        0.01,
                        0.00001,
                    ),
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
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text(
                    "Additional Channels",
                    size="2",
                    weight="bold",
                    color="var(--gray-12)",
                ),
                content=rx.vstack(
                    _additional_channel_row(
                        "Ih (HCN)",
                        AppState.ih_enabled,
                        AppState.set_ih_enabled,
                        AppState.ih_g_max,
                        AppState.set_ih_g_max,
                        "ih_g_max",
                    ),
                    _additional_channel_row(
                        "IKa (A-type K⁺)",
                        AppState.ika_enabled,
                        AppState.set_ika_enabled,
                        AppState.ika_g_max,
                        AppState.set_ika_g_max,
                        "ika_g_max",
                    ),
                    _additional_channel_row(
                        "INaP (Persistent Na⁺)",
                        AppState.inap_enabled,
                        AppState.set_inap_enabled,
                        AppState.inap_g_max,
                        AppState.set_inap_g_max,
                        "inap_g_max",
                    ),
                    _additional_channel_row(
                        "INaR (Resurgent Na⁺)",
                        AppState.inar_enabled,
                        AppState.set_inar_enabled,
                        AppState.inar_g_max,
                        AppState.set_inar_g_max,
                        "inar_g_max",
                    ),
                    _additional_channel_row(
                        "IM (Muscarinic K⁺)",
                        AppState.im_enabled,
                        AppState.set_im_enabled,
                        AppState.im_g_max,
                        AppState.set_im_g_max,
                        "im_g_max",
                    ),
                    _additional_channel_row(
                        "IKir (Inward Rectifier K⁺)",
                        AppState.ikir_enabled,
                        AppState.set_ikir_enabled,
                        AppState.ikir_g_max,
                        AppState.set_ikir_g_max,
                        "ikir_g_max",
                    ),
                    _additional_channel_row(
                        "IKCa (Ca²⁺-activated K⁺)",
                        AppState.ikca_enabled,
                        AppState.set_ikca_enabled,
                        AppState.ikca_g_max,
                        AppState.set_ikca_g_max,
                        "ikca_g_max",
                    ),
                    _additional_channel_row(
                        "ICaL (L-type Ca²⁺)",
                        AppState.ical_enabled,
                        AppState.set_ical_enabled,
                        AppState.ical_g_max,
                        AppState.set_ical_g_max,
                        "ical_g_max",
                    ),
                    _additional_channel_row(
                        "ICaT (T-type Ca²⁺)",
                        AppState.icat_enabled,
                        AppState.set_icat_enabled,
                        AppState.icat_g_max,
                        AppState.set_icat_g_max,
                        "icat_g_max",
                    ),
                    _additional_channel_row(
                        "ICaN (N-type Ca²⁺)",
                        AppState.ican_enabled,
                        AppState.set_ican_enabled,
                        AppState.ican_g_max,
                        AppState.set_ican_g_max,
                        "ican_g_max",
                    ),
                    spacing="2",
                    width="100%",
                ),
                value="additional-channels",
            ),
            collapsible=True,
            variant="ghost",
            width="100%",
        ),
        rx.separator(),
        rx.text("Reversal Potentials", size="2", weight="bold"),
        rx.vstack(
            _reversal_row("E_Na", AppState.E_Na),
            _reversal_row("E_K", AppState.E_K),
            _reversal_row("E_L", AppState.E_L),
            _reversal_row("E_Ca", AppState.E_Ca),
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
    step: float = 1,
) -> rx.Component:
    """Render a single ion concentration row with input."""
    return rx.hstack(
        rx.text(label, size="2", color="gray", width="70px"),
        rx.slider(
            min=min_val,
            max=max_val,
            step=step,
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
            step=str(step),
        ),
        width="100%",
        spacing="2",
    )
