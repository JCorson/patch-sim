"""Reflex configuration for the patch_sim web UI."""

import reflex as rx

config = rx.Config(
    app_name="patch_sim_ui",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
