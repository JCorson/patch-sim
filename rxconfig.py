"""Reflex configuration for the ap_sim web UI."""

import reflex as rx

config = rx.Config(
    app_name="ap_sim_ui",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
