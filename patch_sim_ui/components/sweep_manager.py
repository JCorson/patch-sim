"""Sweep overlay controls and trace visibility popover."""

import reflex as rx

from patch_sim_ui.state import AppState


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
) -> rx.Component:
    """Render a per-channel group: separator, header, current checkbox, gating checkbox.

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

    Returns:
        A fragment containing the conditional channel group.
    """
    return rx.cond(
        enabled_var,
        rx.fragment(
            rx.separator(),
            _section_label(header),
            _trace_checkbox(current_label, current_var, current_handler),
            _trace_checkbox(gating_label, gating_var, gating_handler),
        ),
    )


def _additional_channels_section() -> rx.Component:
    """Render conditional trace groups for all 10 additional channels.

    Each group is only shown when the corresponding channel is enabled.

    Returns:
        A fragment containing all additional channel trace groups.
    """
    return rx.fragment(
        _channel_trace_group(
            "Ih (HCN)",
            AppState.ih_enabled,
            "I_Ih",
            AppState.show_ih_current,
            AppState.set_show_ih_current,
            "Ih gating (r)",
            AppState.show_ih_gating,
            AppState.set_show_ih_gating,
        ),
        _channel_trace_group(
            "IKa (A-type K\u207a)",
            AppState.ika_enabled,
            "I_IKa",
            AppState.show_ika_current,
            AppState.set_show_ika_current,
            "IKa gating (a, b)",
            AppState.show_ika_gating,
            AppState.set_show_ika_gating,
        ),
        _channel_trace_group(
            "INaP (Persistent Na\u207a)",
            AppState.inap_enabled,
            "I_INaP",
            AppState.show_inap_current,
            AppState.set_show_inap_current,
            "INaP gating (p)",
            AppState.show_inap_gating,
            AppState.set_show_inap_gating,
        ),
        _channel_trace_group(
            "INaR (Resurgent Na\u207a)",
            AppState.inar_enabled,
            "I_INaR",
            AppState.show_inar_current,
            AppState.set_show_inar_current,
            "INaR gating (s, hr)",
            AppState.show_inar_gating,
            AppState.set_show_inar_gating,
        ),
        _channel_trace_group(
            "IM (Muscarinic K\u207a)",
            AppState.im_enabled,
            "I_IM",
            AppState.show_im_current,
            AppState.set_show_im_current,
            "IM gating (w)",
            AppState.show_im_gating,
            AppState.set_show_im_gating,
        ),
        _channel_trace_group(
            "IKir (Inward Rectifier K\u207a)",
            AppState.ikir_enabled,
            "I_IKir",
            AppState.show_ikir_current,
            AppState.set_show_ikir_current,
            "IKir gating (kir)",
            AppState.show_ikir_gating,
            AppState.set_show_ikir_gating,
        ),
        _channel_trace_group(
            "IKCa (Ca\u00b2\u207a-activated K\u207a)",
            AppState.ikca_enabled,
            "I_IKCa",
            AppState.show_ikca_current,
            AppState.set_show_ikca_current,
            "IKCa gating (q)",
            AppState.show_ikca_gating,
            AppState.set_show_ikca_gating,
        ),
        _channel_trace_group(
            "ICaL (L-type Ca\u00b2\u207a)",
            AppState.ical_enabled,
            "I_ICaL",
            AppState.show_ical_current,
            AppState.set_show_ical_current,
            "ICaL gating (d, f)",
            AppState.show_ical_gating,
            AppState.set_show_ical_gating,
        ),
        _channel_trace_group(
            "ICaT (T-type Ca\u00b2\u207a)",
            AppState.icat_enabled,
            "I_ICaT",
            AppState.show_icat_current,
            AppState.set_show_icat_current,
            "ICaT gating (dt, ft)",
            AppState.show_icat_gating,
            AppState.set_show_icat_gating,
        ),
        _channel_trace_group(
            "ICaN (N-type Ca\u00b2\u207a)",
            AppState.ican_enabled,
            "I_ICaN",
            AppState.show_ican_current,
            AppState.set_show_ican_current,
            "ICaN gating (dn, fn)",
            AppState.show_ican_gating,
            AppState.set_show_ican_gating,
        ),
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
        _trace_checkbox("Voltage", AppState.show_voltage, AppState.set_show_voltage),
        rx.separator(),
        _section_label("Gating Variables"),
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
        _additional_channels_section(),
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
        rx.separator(),
        _section_label("Gating Variables"),
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
                AppState.clamp_mode == "Current Clamp",
                _cc_popover_content(),
                _vc_popover_content(),
            ),
        ),
    )


def _sweep_chip(sweep) -> rx.Component:
    """Render a colour-coded badge for a saved sweep.

    Args:
        sweep: A ``Sweep`` instance from the saved sweeps list.

    Returns:
        A badge component showing the sweep colour dot and label.
    """
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
    """Trace visibility popover and sweep overlay management bar.

    Returns:
        A vstack containing the Traces popover button and the sweep
        management controls (saved sweep badges, Add sweep, Clear).
    """
    return rx.vstack(
        rx.hstack(
            _trace_visibility_popover(),
            rx.separator(orientation="vertical"),
            rx.button(
                rx.icon("scroll-text"),
                "Logs",
                on_click=AppState.toggle_log_panel,
                size="1",
                variant="soft",
            ),
            rx.separator(orientation="vertical"),
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
