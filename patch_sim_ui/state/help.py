"""Help page state for the patch_sim web UI."""

import reflex as rx

from patch_sim_ui.docs_loader import DEFAULT_TOPIC, PROSE


class HelpState(rx.State):
    """State for the ``/help`` documentation page."""

    topic: str = DEFAULT_TOPIC

    @rx.var
    def current_body(self) -> str:
        """Markdown body for the selected help topic.

        Returns:
            The prose for ``self.topic``, falling back to the default topic
            if the slug is unknown.
        """
        return PROSE.get(self.topic, PROSE[DEFAULT_TOPIC])

    def set_topic(self, topic: str) -> None:
        """Select the help topic to display.

        Args:
            topic: Slug of the topic to show (a key of the loaded prose).
        """
        self.topic = topic
