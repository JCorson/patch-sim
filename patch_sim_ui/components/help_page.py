"""The ``/help`` documentation page: topic nav plus rendered Markdown."""

import reflex as rx

from patch_sim_ui.docs_loader import API_DOCS_URL, HELP_TOPICS
from patch_sim_ui.state.help import HelpState


def _topic_button(slug: str, label: str) -> rx.Component:
    """Return one topic nav button that selects ``slug`` when clicked.

    Args:
        slug: Topic slug passed to :meth:`HelpState.set_topic`.
        label: Human-readable button text.

    Returns:
        A full-width button highlighted when its topic is active.
    """
    return rx.button(
        label,
        on_click=HelpState.set_topic(slug),
        variant=rx.cond(HelpState.topic == slug, "solid", "soft"),
        color_scheme="blue",
        width="100%",
        justify_content="start",
        size="2",
    )


def _help_header() -> rx.Component:
    """Top bar with a back link, title, and link to the API reference site.

    Returns:
        The help page header component.
    """
    return rx.hstack(
        rx.link(
            rx.button(
                rx.icon("arrow-left"),
                "Back to simulator",
                color_scheme="gray",
                variant="soft",
                size="2",
            ),
            href="/",
        ),
        rx.spacer(),
        rx.heading("Help & Documentation", size="6"),
        rx.spacer(),
        rx.link(
            rx.button(
                rx.icon("external-link"),
                "API reference",
                variant="soft",
                size="2",
            ),
            href=API_DOCS_URL,
            is_external=True,
        ),
        rx.color_mode.button(allow_system=True, variant="ghost", size="2"),
        width="100%",
        padding_x="4",
        padding_y="3",
        border_bottom="1px solid var(--gray-4)",
        align="center",
    )


def _topic_nav() -> rx.Component:
    """Left rail listing the help topics.

    Returns:
        A sidebar of topic buttons.
    """
    return rx.box(
        rx.vstack(
            rx.text("Topics", size="1", weight="bold", color="gray"),
            *[_topic_button(slug, label) for slug, label in HELP_TOPICS],
            spacing="2",
            width="100%",
        ),
        width="240px",
        min_width="240px",
        padding="4",
        border_right="1px solid var(--gray-4)",
        background="var(--gray-1)",
        height="calc(100vh - 60px)",
    )


def help_page() -> rx.Component:
    """Root component for the ``/help`` route.

    Returns:
        A full-page layout: header, topic nav, and the rendered Markdown body
        for the selected topic.
    """
    return rx.vstack(
        _help_header(),
        rx.hstack(
            _topic_nav(),
            rx.scroll_area(
                rx.box(
                    rx.markdown(HelpState.current_body),
                    max_width="820px",
                    padding_x="6",
                    padding_y="4",
                ),
                height="calc(100vh - 60px)",
                width="100%",
            ),
            spacing="0",
            width="100%",
            align="start",
        ),
        spacing="0",
        width="100%",
        min_height="100vh",
    )
