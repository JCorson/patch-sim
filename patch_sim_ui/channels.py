"""Ion channel registry for the patch_sim UI.

Each :class:`ChannelMeta` entry in :data:`CHANNELS` is the single source of
truth for one channel — its label, field names, gating variables, slider
range, and display colours.  All other modules derive their per-channel
lists and dicts from this registry rather than maintaining parallel
enumerations.

Adding a new channel requires only a single new :class:`ChannelMeta` entry
here; all downstream lists and dicts are recomputed automatically.
"""

from dataclasses import dataclass


@dataclass
class ChannelMeta:
    """Metadata for a single ion channel.

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
    def column_name(self) -> str:
        """Simulation result column name for this channel's current.

        Mirrors :attr:`patch_sim.channels.IonChannel.current_name` — the
        ``f"I{IonChannel.name}"`` convention used throughout the simulator.
        Aux registry entries already encode the leading ``I`` (e.g.
        ``current_key="Ih"`` for ``IonChannel.name="h"``); HH-classic core
        entries store the bare channel name (e.g. ``current_key="Na"``) and
        get the ``I`` prefix added here.

        Returns:
            Simulation column name, e.g. ``"INa"`` or ``"Ih"``.
        """
        if self.current_key.startswith("I") and self.current_key != "I":
            return self.current_key
        return f"I{self.current_key}"

    @property
    def gating_label(self) -> str:
        """Visibility checkbox label for the gating variable trace(s).

        Returns:
            Label string, e.g. ``"Ih gating (r)"``.
        """
        return f"{self.current_key} gating ({', '.join(self.gating_vars)})"


#: Ordered registry of every ion channel exposed in the UI.
#:
#: Single source of truth for per-channel metadata; downstream dicts and
#: field lists are derived from this tuple so that adding a channel
#: requires only one new entry here.
CHANNELS: tuple[ChannelMeta, ...] = (
    # HH-classic core channels (per-preset kinetics; the slider only edits g_max).
    ChannelMeta(
        id="na",
        current_key="Na",
        label="Na (fast Na⁺)",
        gating_vars=("m", "h"),
        g_max_range=(0.0, 300.0, 1.0),
        current_color="#ff7f0e",
        gating_var_colors={"m": "#ff7f0e", "h": "#2ca02c"},
    ),
    ChannelMeta(
        id="k",
        current_key="K",
        label="K (delayed rectifier)",
        gating_vars=("n",),
        g_max_range=(0.0, 100.0, 0.5),
        current_color="#2ca02c",
        gating_var_colors={"n": "#1f77b4"},
    ),
    ChannelMeta(
        id="nal",
        current_key="NaL",
        label="NaL (Na⁺ leak)",
        gating_vars=(),
        g_max_range=(0.0, 2.0, 0.01),
        current_color="#7f7f7f",
        gating_var_colors={},
    ),
    ChannelMeta(
        id="kl",
        current_key="KL",
        label="KL (K⁺ leak)",
        gating_vars=(),
        g_max_range=(0.0, 2.0, 0.01),
        current_color="#bcbd22",
        gating_var_colors={},
    ),
    # Auxiliary channels.
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

#: Channel ids whose gating variables are exposed via the per-gate visibility
#: fields (``show_potassium_activation`` for ``n``, ``show_sodium_activation``
#: for ``m``, ``show_sodium_inactivation`` for ``h``) rather than the joint
#: ``show_<id>_gating`` field used for every other channel.  ``nal`` and
#: ``kl`` are leak channels with no gating variables, included for completeness
#: so callers can use a single set when they want to skip the four HH-classic
#: core channels.
HH_CLASSIC_CHANNEL_IDS: frozenset[str] = frozenset({"na", "k", "nal", "kl"})

#: Maps simulation result column names (e.g. ``"INa"``, ``"Ih"``) to their
#: ``show_*_current`` visibility field names.  Used by VisibilityState and
#: SimulationState to translate per-channel current keys into the
#: corresponding bool flag.
CURRENT_FIELD_MAP: dict[str, str] = {
    ch.column_name: ch.current_visibility_field for ch in CHANNELS
}

#: Maps gating variable names to their ``show_*_gating`` visibility field
#: names.  HH-classic core channels are excluded — their gates ``n``, ``m``,
#: ``h`` are wired to the per-gate fields above.
GATING_FIELD_MAP: dict[str, str] = {
    gv: ch.gating_visibility_field
    for ch in CHANNELS
    if ch.id not in HH_CLASSIC_CHANNEL_IDS
    for gv in ch.gating_vars
}
