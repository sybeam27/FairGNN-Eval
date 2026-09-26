"""The shared tone of every paper figure, taken from Fig. 2 (fig2_sign_resolution).

    text      serif (STIX), 8 pt body, 7.5 pt ticks, black labels; grey axes (#52514e), no top/right spines
    fills     a light tint of the role colour (TINT) with a thin outline in the full colour
    resolved  a stronger tint (TINT_RES) and a star after the value; unresolved keeps the light tint
    legend    thin, light box (#d0cfca, 0.4 pt), square corners
"""
import matplotlib
import matplotlib.pyplot as plt

INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9993"

# One palette for the whole paper: a colour always means the same ROLE, never a method.
# Method families are told apart by marker shape only (square / triangle / circle).
ROLE = {
    "intervention": "#8c3b6b",   # plum   : M^{+I} side -- tau_{-I->+I}, +fair, E_{+I}
    "surrounding": "#3b6b8c",    # slate  : M^{-I} side -- tau_{B->-I}, +prop, E_{-I}
    "baseline_step": "#2a7f7f",  # teal   : B -> M^base (the step that leaves the common baseline)
    "level": "#3d3c39",          # ink    : levels and totals -- tau_{B->+I}, pkg, tau_cap / tau_full
    "run_noise": MUTED,          # grey   : prefix / run difference (R_traj)
    "sign_flip": "#6b4fa8",      # violet : the point estimate changes sign
    "lost": "#b03a2e",           # brick  : resolved -> unresolved
    "gained": "#c99a1e",         # ochre  : unresolved -> resolved
}
LEGEND_EDGE = "#d0cfca"
TINT, TINT_RES = 0.82, 0.55          # fill = colour mixed with white by this fraction
OUTLINE = 0.8                        # bar outline width, pt

RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.labelcolor": INK, "text.color": INK,
    # serif text with matching math, so the figures read like the paper body (STIX ~ Times)
    "font.family": "serif", "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply_tone():
    plt.rcParams.update(RC)


def tint(c, t=TINT):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def box(leg):
    """the Fig. 2 legend frame"""
    fr = leg.get_frame()
    fr.set_linewidth(0.4)
    fr.set_edgecolor(LEGEND_EDGE)
    return leg
