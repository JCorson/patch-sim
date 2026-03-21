"""Reflex configuration for the patch_sim web UI."""

import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="patch_sim_ui",
    disable_plugins=[SitemapPlugin],
)
