"""Neuron parameter panel component."""

import reflex as rx

from patch_sim.presets import NEURON_PRESET_NAMES
from patch_sim_ui.channels import ADDITIONAL_CHANNELS
from patch_sim_ui.constants import PARAM_RANGES
from patch_sim_ui.state.neuron import NeuronState


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


def _ion_row(
    label: str,
    var: rx.Var,
    handler,
    min_val: float,
    max_val: float,
    step: float = 1,
) -> rx.Component:
    """Render a single ion concentration row with a slider and numeric input.

    Args:
        label: Ion species label (e.g. ``"Na⁺ out"``).
        var: Reactive state variable bound to the slider and input.
        handler: Event handler called when the value changes.
        min_val: Minimum allowed concentration (mM).
        max_val: Maximum allowed concentration (mM).
        step: Slider and input increment step (mM).

    Returns:
        An hstack containing a label, slider, and numeric input.
    """
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


def _reversal_str(label: str, value: rx.Var, unit: str = "mV") -> rx.Component:
    """Render a read-only reversal potential display row.

    Args:
        label: Ion species label (e.g. ``"E_Na"``).
        value: Reactive float var holding the computed reversal potential.
        unit: Unit string appended after the value (default ``"mV"``).

    Returns:
        An hstack showing the label, formatted value, and unit.
    """
    return rx.hstack(
        rx.text(label, size="2", color="gray"),
        rx.text(f"{value:.2f}", size="2"),
        rx.text(f" {unit}", size="1", color="gray"),
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


# (label, enabled_var, enabled_setter, g_var, g_setter, param_key)
# Derived from the channel registry in patch_sim_ui.channels.
_ADDITIONAL_CHANNEL_ROW_SPECS = [
    (
        ch.label,
        getattr(NeuronState, ch.enabled_field),
        getattr(NeuronState, f"set_{ch.enabled_field}"),
        getattr(NeuronState, ch.g_max_field),
        getattr(NeuronState, f"set_{ch.g_max_field}"),
        ch.g_max_field,
    )
    for ch in ADDITIONAL_CHANNELS
]


def neuron_panel() -> rx.Component:
    """Sidebar panel for configuring Hodgkin-Huxley neuron parameters."""
    return rx.vstack(
        rx.heading("Neuron Parameters", size="4"),
        rx.select(
            NEURON_PRESET_NAMES,
            placeholder="Load neuron type…",
            value=NeuronState.active_neuron_type,
            on_change=NeuronState.load_neuron_preset,
            width="100%",
            size="2",
        ),
        rx.separator(),
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text(
                    "Conductances (mS/cm²)",
                    size="2",
                    weight="bold",
                    color="var(--gray-12)",
                ),
                content=rx.vstack(
                    _param_row(
                        "g_Na",
                        NeuronState.g_Na,
                        NeuronState.set_g_Na,
                        *PARAM_RANGES["g_Na"],
                    ),
                    _param_row(
                        "g_K",
                        NeuronState.g_K,
                        NeuronState.set_g_K,
                        *PARAM_RANGES["g_K"],
                    ),
                    _param_row(
                        "g_NaL",
                        NeuronState.g_NaL,
                        NeuronState.set_g_NaL,
                        *PARAM_RANGES["g_NaL"],
                    ),
                    _param_row(
                        "g_KL",
                        NeuronState.g_KL,
                        NeuronState.set_g_KL,
                        *PARAM_RANGES["g_KL"],
                    ),
                    spacing="1",
                    width="100%",
                ),
                value="conductances",
            ),
            collapsible=True,
            variant="ghost",
            width="100%",
        ),
        rx.separator(),
        rx.accordion.root(
            rx.accordion.item(
                header=rx.text(
                    "Membrane Properties",
                    size="2",
                    weight="bold",
                    color="var(--gray-12)",
                ),
                content=rx.vstack(
                    _param_row(
                        "C_m (µF/cm²)",
                        NeuronState.C_m,
                        NeuronState.set_C_m,
                        *PARAM_RANGES["C_m"],
                    ),
                    _param_row(
                        "v_rest (mV)",
                        NeuronState.v_rest,
                        NeuronState.set_v_rest,
                        *PARAM_RANGES["v_rest"],
                    ),
                    _param_row(
                        "T (K)",
                        NeuronState.T,
                        NeuronState.set_T,
                        *PARAM_RANGES["T"],
                    ),
                    rx.tooltip(
                        rx.hstack(
                            rx.checkbox(
                                checked=NeuronState.has_area_cm2,
                                on_change=NeuronState.set_has_area_cm2,
                            ),
                            rx.text(
                                "Use cell area (report MΩ, pF)",
                                size="2",
                                color="gray",
                            ),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        content=(
                            "When enabled, passive properties are reported in "
                            "absolute units (MΩ, pF) using the cell surface "
                            "area below. Off: per-area density units "
                            "(kΩ·cm², µF/cm²). Examples: FS interneuron ≈ "
                            "3e-6 cm², cortical pyramidal ≈ 2e-5 cm², "
                            "Purkinje ≈ 2.5e-4 cm²."
                        ),
                    ),
                    rx.cond(
                        NeuronState.has_area_cm2,
                        _param_row(
                            "Area (cm²)",
                            NeuronState.area_cm2,
                            NeuronState.set_area_cm2,
                            *PARAM_RANGES["area_cm2"],
                        ),
                        rx.box(),
                    ),
                    spacing="1",
                    width="100%",
                ),
                value="membrane-properties",
            ),
            collapsible=True,
            variant="ghost",
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
                    _ion_row(
                        "Na⁺ out",
                        NeuronState.Na_out,
                        NeuronState.set_Na_out,
                        *PARAM_RANGES["Na_out"],
                    ),
                    _ion_row(
                        "Na⁺ in",
                        NeuronState.Na_in,
                        NeuronState.set_Na_in,
                        *PARAM_RANGES["Na_in"],
                    ),
                    _ion_row(
                        "K⁺ out",
                        NeuronState.K_out,
                        NeuronState.set_K_out,
                        *PARAM_RANGES["K_out"],
                    ),
                    _ion_row(
                        "K⁺ in",
                        NeuronState.K_in,
                        NeuronState.set_K_in,
                        *PARAM_RANGES["K_in"],
                    ),
                    _ion_row(
                        "Ca²⁺ out",
                        NeuronState.Ca_out,
                        NeuronState.set_Ca_out,
                        *PARAM_RANGES["Ca_out"],
                    ),
                    _ion_row(
                        "Ca²⁺ in",
                        NeuronState.Ca_in,
                        NeuronState.set_Ca_in,
                        *PARAM_RANGES["Ca_in"],
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
                    *[
                        _additional_channel_row(*spec)
                        for spec in _ADDITIONAL_CHANNEL_ROW_SPECS
                    ],
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
        rx.grid(
            _reversal_str("E_Na", NeuronState.E_Na),
            _reversal_str("E_K", NeuronState.E_K),
            _reversal_str("E_L", NeuronState.E_L),
            _reversal_str("E_Ca", NeuronState.E_Ca),
            spacing="1",
            width="100%",
            columns="2",
        ),
        spacing="3",
        width="100%",
        padding="4",
    )
