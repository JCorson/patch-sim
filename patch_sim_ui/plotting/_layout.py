"""Shared layout constants for analysis figures.

Private to the :mod:`patch_sim_ui.plotting` package.  Holds the one layout
preset every analysis builder (F-I, I-V, g-V, SFA, phase-plane,
hyperpolarization, tau-V) applies via
``fig.update_layout(**_ANALYSIS_FIGURE_LAYOUT, ...)``.
"""

# Shared layout kwargs applied to every analysis sub-figure (F-I, I-V, g-V,
# SFA, phase-plane).  Kept separate from the main trace plot because the
# analysis figures are smaller and have tighter margins.
_ANALYSIS_FIGURE_LAYOUT: dict = {
    "template": "plotly_white",
    "margin": {"l": 50, "r": 10, "t": 10, "b": 40},
    "height": 260,
}
