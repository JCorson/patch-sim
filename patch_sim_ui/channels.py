"""Additional ion channel registry for the patch_sim UI.

Each :class:`ChannelMeta` entry in :data:`ADDITIONAL_CHANNELS` is the single
source of truth for one auxiliary channel — its label, field names, gating
variables, slider range, and display colours.  All other modules derive their
per-channel lists and dicts from this registry rather than maintaining
parallel enumerations.

Adding a new channel requires only a single new :class:`ChannelMeta` entry
here; all downstream lists and dicts are recomputed automatically.
"""

from dataclasses import dataclass


@dataclass
class ChannelMeta:
    """Metadata for a single additional ion channel.

    Attributes:
        id: Lowercase identifier used as a field prefix, e.g. ``"ih"``.
        current_key: Simulation result field name for the current,
            e.g. ``"Ih"``.
        label: Display name shown in the UI panels, e.g. ``"Ih (HCN)"``.
        gating_vars: Gating variable names produced by this channel's model.
        g_max_range: ``(min, max, step)`` for the conductance slider (mS/cm²).
        current_color: Hex colour string for the current trace in the plot.
        gating_var_colors: Mapping of gating variable name to hex colour.
    """

    id: str
    current_key: str
    label: str
    gating_vars: tuple[str, ...]
    g_max_range: tuple[float, float, float]
    current_color: str
    gating_var_colors: dict[str, str]

    @property
    def g_max_field(self) -> str:
        """NeuronState attribute name for the conductance slider.

        Returns:
            Field name, e.g. ``"ih_g_max"``.
        """
        return f"{self.id}_g_max"

    @property
    def current_visibility_field(self) -> str:
        """VisibilityState field for current trace visibility.

        Returns:
            Field name, e.g. ``"show_ih_current"``.
        """
        return f"show_{self.id}_current"

    @property
    def gating_visibility_field(self) -> str:
        """VisibilityState field for gating trace visibility.

        Returns:
            Field name, e.g. ``"show_ih_gating"``.
        """
        return f"show_{self.id}_gating"

    @property
    def current_label(self) -> str:
        """Visibility checkbox label for the current trace.

        Returns:
            Label string, e.g. ``"I_Ih"``.
        """
        return f"I_{self.current_key}"

    @property
    def gating_label(self) -> str:
        """Visibility checkbox label for the gating variable trace(s).

        Returns:
            Label string, e.g. ``"Ih gating (r)"``.
        """
        return f"{self.current_key} gating ({', '.join(self.gating_vars)})"


#: Ordered registry of all additional (non-HH-classic) ion channels.
#:
#: This is the single source of truth for per-channel metadata.  All
#: downstream dicts and field lists are derived from this tuple so that
#: adding a channel requires only one new entry here.
ADDITIONAL_CHANNELS: tuple[ChannelMeta, ...] = (
    ChannelMeta(
        id="ih",
        current_key="Ih",
        label="Ih (HCN)",
        gating_vars=("r",),
        g_max_range=(0.0, 1.0, 0.01),
        current_color="#d62728",
        gating_var_colors={"r": "#bcbd22"},
    ),
    ChannelMeta(
        id="ika",
        current_key="IKa",
        label="IKa (A-type K\u207a)",
        gating_vars=("a", "b"),
        g_max_range=(0.0, 100.0, 0.1),
        current_color="#9467bd",
        gating_var_colors={"a": "#8c564b", "b": "#e377c2"},
    ),
    ChannelMeta(
        id="ikv31",
        current_key="IKv31",
        label="IKv31 (Kv3.1-type K\u207a)",
        gating_vars=("nk",),
        g_max_range=(0.0, 100.0, 0.5),
        current_color="#DAA520",
        gating_var_colors={"nk": "#DAA520"},
    ),
    ChannelMeta(
        id="mskv",
        current_key="Kv",
        label="Kv (Mainen-Sejnowski K\u207a)",
        gating_vars=("nKv",),
        g_max_range=(0.0, 50.0, 0.1),
        current_color="#1f77b4",
        gating_var_colors={"nKv": "#aec7e8"},
    ),
    ChannelMeta(
        id="inap",
        current_key="INaP",
        label="INaP (Persistent Na\u207a)",
        gating_vars=("p",),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#e377c2",
        gating_var_colors={"p": "#7f7f7f"},
    ),
    ChannelMeta(
        id="inar",
        current_key="INaR",
        label="INaR (Resurgent Na\u207a)",
        gating_vars=("s", "hr"),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#bcbd22",
        gating_var_colors={"s": "#d62728", "hr": "#9467bd"},
    ),
    ChannelMeta(
        id="im",
        current_key="IM",
        label="IM (Muscarinic K\u207a)",
        gating_vars=("w",),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#17becf",
        gating_var_colors={"w": "#17becf"},
    ),
    ChannelMeta(
        id="katp",
        current_key="IKATP",
        label="K_ATP (ATP-sensitive K\u207a)",
        gating_vars=("kATP",),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#bcbd22",
        gating_var_colors={"kATP": "#bcbd22"},
    ),
    ChannelMeta(
        id="ikir",
        current_key="IKir",
        label="IKir (Inward Rectifier K\u207a)",
        gating_vars=("kir",),
        g_max_range=(0.0, 2.0, 0.01),
        current_color="#8c564b",
        gating_var_colors={"kir": "#aec7e8"},
    ),
    ChannelMeta(
        id="ikca",
        current_key="IKCa",
        label="IKCa (Ca\u00b2\u207a-activated K\u207a)",
        gating_vars=("q",),
        g_max_range=(0.0, 10.0, 0.1),
        current_color="#ff9896",
        gating_var_colors={"q": "#ffbb78"},
    ),
    ChannelMeta(
        id="ical",
        current_key="ICaL",
        label="ICaL (L-type Ca\u00b2\u207a)",
        gating_vars=("d", "f"),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#aec7e8",
        gating_var_colors={"d": "#98df8a", "f": "#ff9896"},
    ),
    ChannelMeta(
        id="cav13",
        current_key="Cav1.3",
        label="Cav1.3 (LVA L-type Ca\u00b2\u207a)",
        gating_vars=("dL13", "fL13"),
        g_max_range=(0.0, 1.0, 0.01),
        current_color="#5fb3d8",
        gating_var_colors={"dL13": "#3a86a3", "fL13": "#9ed8eb"},
    ),
    ChannelMeta(
        id="icat",
        current_key="ICaT",
        label="ICaT (T-type Ca\u00b2\u207a)",
        gating_vars=("dt", "ft"),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#98df8a",
        gating_var_colors={"dt": "#c5b0d5", "ft": "#c49c94"},
    ),
    ChannelMeta(
        id="ican",
        current_key="ICaN",
        label="ICaN (N-type Ca\u00b2\u207a)",
        gating_vars=("dn", "fn"),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#c5b0d5",
        gating_var_colors={"dn": "#f7b6d2", "fn": "#dbdb8d"},
    ),
    ChannelMeta(
        id="sk",
        current_key="SK",
        label="SK (Ca\u00b2\u207a-activated K\u207a)",
        gating_vars=("qSK",),
        g_max_range=(0.0, 5.0, 0.01),
        current_color="#ff7f50",
        gating_var_colors={"qSK": "#ffa07a"},
    ),
)

#: Maps additional-channel current keys to their show_* visibility field names.
#: Used by VisibilityState and SimulationState.
ADDITIONAL_CURRENT_FIELD_MAP: dict[str, str] = {
    ch.current_key: ch.current_visibility_field for ch in ADDITIONAL_CHANNELS
}

#: Maps additional-channel gating variable names to their show_* visibility field names.
#: Used by VisibilityState and SimulationState.
ADDITIONAL_GATING_FIELD_MAP: dict[str, str] = {
    gv: ch.gating_visibility_field
    for ch in ADDITIONAL_CHANNELS
    for gv in ch.gating_vars
}
