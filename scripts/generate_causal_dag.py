"""
CVPR/NeurIPS-style vector figures for manuscript:
1) Fig11_Causal_DAG
2) Fig12_Method_Architecture
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# Soft, publication-friendly palette
PALETTE = {
    "data": "#EAF2FF",
    "process": "#EEF7F1",
    "model": "#F4EDFF",
    "eval": "#FFF4E8",
    "accent": "#6E83B7",
    "edge": "#425466",
    "muted": "#708090",
}


def _round_box(ax, x, y, w, h, text, fill, edge="#425466", fs=9, lw=1.2):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#1F2D3A")


def _arrow(ax, p1, p2, lw=1.3, rad=0.0, style="-|>", ls="-", color="#425466"):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def _icon_dot(ax, x, y, color):
    ax.add_patch(Circle((x, y), 0.04, color=color, ec="none", zorder=3))


def draw_causal_dag():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Canonical causal layout (inspired by top-paper DAG conventions)
    _round_box(ax, 1.5, 4.9, 2.2, 0.78, "Observed Covariates\n$X$ (k1/k2, SOFA, labs)", PALETTE["data"])
    _round_box(ax, 4.2, 4.9, 1.9, 0.78, "Treatment\n$T$", PALETTE["model"])
    _round_box(ax, 6.8, 4.9, 2.2, 0.78, "Outcome\n$Y$ (RRT in 24h)", PALETTE["eval"])
    _round_box(ax, 4.2, 3.45, 2.3, 0.78, "Propensity Score\n$e(X)=P(T=1|X)$", PALETTE["process"])
    _round_box(ax, 6.8, 3.45, 2.4, 0.78, "Prediction Head\n$\\hat{p}=P(Y=1|X,T)$", PALETTE["process"])
    _round_box(ax, 9.0, 3.45, 1.7, 0.78, "Decision\n$\\hat{y}=\\mathbb{1}[\\hat{p}\\geq \\tau]$", PALETTE["eval"], fs=8.5)

    # Latent confounder U (circle)
    u = Circle((2.8, 2.0), 0.38, linewidth=1.2, edgecolor="#425466", facecolor="#F7F7F7")
    ax.add_patch(u)
    ax.text(2.8, 2.0, "U", ha="center", va="center", fontsize=10, color="#2B3E50")
    ax.text(2.8, 1.45, "Latent confounder", ha="center", va="center", fontsize=8, color="#566B7F")

    # Core causal edges
    _arrow(ax, (2.6, 4.9), (3.25, 4.9))   # X -> T
    _arrow(ax, (2.6, 4.75), (5.7, 4.95))   # X -> Y
    _arrow(ax, (5.15, 4.9), (5.85, 4.9))   # T -> Y
    _arrow(ax, (1.9, 4.5), (3.3, 3.75))    # X -> e(X)
    _arrow(ax, (4.2, 4.5), (4.2, 3.85))    # T -> e(X)
    _arrow(ax, (5.0, 3.45), (5.6, 3.45))   # e(X) -> pred (IPW weighting path)
    _arrow(ax, (6.8, 4.5), (6.8, 3.85))    # Y signal path to pred target
    _arrow(ax, (8.0, 3.45), (8.15, 3.45))  # pred -> decision

    # Confounding edges (dashed)
    _arrow(ax, (3.05, 2.28), (4.0, 4.5), ls="--", color=PALETTE["muted"])  # U -> T
    _arrow(ax, (3.15, 2.2), (6.5, 4.5), ls="--", color=PALETTE["muted"])   # U -> Y

    # Adjustment note
    ax.text(5.9, 2.1, "Backdoor adjustment via $X$ and IPW with $e(X)$", fontsize=8.5, color="#5A6D7F")

    # Minimal legend
    _icon_dot(ax, 0.9, 0.72, PALETTE["data"])
    ax.text(1.05, 0.72, "Observed", fontsize=8.5, va="center", color="#2E3B4E")
    _icon_dot(ax, 2.1, 0.72, "#F7F7F7")
    ax.text(2.25, 0.72, "Latent", fontsize=8.5, va="center", color="#2E3B4E")
    ax.plot([3.15, 3.65], [0.72, 0.72], "--", color=PALETTE["muted"], lw=1.2)
    ax.text(3.75, 0.72, "Confounding path", fontsize=8.5, va="center", color="#2E3B4E")

    ax.set_title("Fig11. Causal DAG (Top-Conference Style)", fontsize=12.5, color="#223447", pad=10)

    fig.savefig(os.path.join(FIG_DIR, "Fig11_Causal_DAG_Generated.png"), dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Fig11_Causal_DAG_Generated.pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_causal_dag_minimal():
    """
    Minimal NeurIPS-style DAG for main paper.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # Main nodes
    _round_box(ax, 1.3, 3.4, 2.0, 0.74, "Covariates\n$X$", PALETTE["data"])
    _round_box(ax, 3.8, 3.4, 1.5, 0.74, "Treatment\n$T$", PALETTE["model"])
    _round_box(ax, 6.2, 3.4, 1.7, 0.74, "Outcome\n$Y$", PALETTE["eval"])
    _round_box(ax, 3.8, 2.0, 2.1, 0.74, "Propensity\n$e(X)$", PALETTE["process"])
    _round_box(ax, 6.2, 2.0, 2.4, 0.74, "Predictor\n$\\hat{p}=P(Y=1|X,T)$", PALETTE["process"], fs=8.8)
    _round_box(ax, 8.0, 2.0, 1.5, 0.74, "Decision\n$\\hat{y}$", PALETTE["eval"])

    # Latent confounder
    u = Circle((2.6, 1.0), 0.30, linewidth=1.1, edgecolor=PALETTE["edge"], facecolor="#F7F7F7")
    ax.add_patch(u)
    ax.text(2.6, 1.0, "U", ha="center", va="center", fontsize=9.5, color="#2B3E50")

    # Core edges
    _arrow(ax, (2.3, 3.4), (3.02, 3.4))  # X->T
    _arrow(ax, (4.55, 3.4), (5.32, 3.4))  # T->Y
    _arrow(ax, (2.35, 3.18), (5.35, 3.30), rad=0.02)  # X->Y
    _arrow(ax, (1.9, 3.05), (3.0, 2.3))  # X->e
    _arrow(ax, (3.8, 3.03), (3.8, 2.37))  # T->e
    _arrow(ax, (4.85, 2.0), (5.0, 2.0))  # e->pred
    _arrow(ax, (7.4, 2.0), (7.25, 2.0))  # pred->decision (short overlap for compactness)

    # Confounding edges
    _arrow(ax, (2.83, 1.22), (3.55, 3.03), ls="--", color=PALETTE["muted"])
    _arrow(ax, (2.88, 1.18), (5.95, 3.02), ls="--", color=PALETTE["muted"])

    # Tiny note
    ax.text(5.0, 0.55, "Adjusted by $X$ and IPW with $e(X)$", fontsize=8.5, color="#5A6D7F", ha="center")

    ax.set_title("Fig11. Minimal Causal DAG (Main-text Version)", fontsize=12, color="#223447", pad=8)
    fig.savefig(os.path.join(FIG_DIR, "Fig11_Causal_DAG_Main.png"), dpi=450, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Fig11_Causal_DAG_Main.pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_method_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Stage 1: data sources
    _round_box(ax, 1.2, 4.8, 2.1, 0.9, "MIMIC-IV\nPreprocessed", PALETTE["data"])
    _round_box(ax, 1.2, 3.4, 2.1, 0.9, "eICU\nPreprocessed", PALETTE["data"])
    _round_box(ax, 3.5, 4.1, 2.6, 1.2, "Data Fusion + Split\nExternal holdout: 50% eICU\nFusion split: 7:2:1", PALETTE["process"], fs=8.8)

    # Stage 2: feature module
    _round_box(ax, 6.0, 4.8, 2.6, 0.9, "Feature Engineering\n38 features (Excl. PS)", PALETTE["process"])
    _round_box(ax, 6.0, 3.3, 2.6, 0.9, "Causal Adjustment\nIPW Weights: $w_i$", PALETTE["model"])

    # Stage 3: model
    _round_box(ax, 8.7, 4.1, 2.8, 1.5, "Regularized Stacking\nLGB + XGB + RF -> LR\nLoss IPW Weighted", PALETTE["model"], fs=8.8)

    # Stage 4: selection + outputs
    _round_box(ax, 10.9, 4.8, 2.0, 0.9, "Validation Tuning\n$\\tau=0.57$", PALETTE["eval"])
    _round_box(ax, 10.9, 3.4, 2.0, 0.9, "Evaluation\nAUC/AP/Brier", PALETTE["eval"])
    _round_box(ax, 10.9, 2.1, 2.0, 0.9, "Interpretability\nSHAP + DCA", PALETTE["eval"])

    # Arrows
    _arrow(ax, (2.25, 4.8), (2.85, 4.35))
    _arrow(ax, (2.25, 3.4), (2.85, 3.85))
    _arrow(ax, (4.8, 4.35), (5.0, 4.65))
    _arrow(ax, (4.8, 3.85), (5.0, 3.45))
    _arrow(ax, (7.3, 4.8), (7.8, 4.35))
    _arrow(ax, (7.3, 3.3), (7.8, 3.95))
    _arrow(ax, (9.95, 4.35), (9.9, 4.75))
    _arrow(ax, (9.95, 4.05), (9.9, 3.45))
    _arrow(ax, (10.9, 3.0), (10.9, 2.55))

    # Subtle section headers
    ax.text(1.2, 5.55, "Data", fontsize=9, color=PALETTE["accent"], ha="center")
    ax.text(6.0, 5.55, "Feature + Causal Module", fontsize=9, color=PALETTE["accent"], ha="center")
    ax.text(8.7, 5.55, "Model", fontsize=9, color=PALETTE["accent"], ha="center")
    ax.text(10.9, 5.55, "Outputs", fontsize=9, color=PALETTE["accent"], ha="center")

    ax.set_title("Fig12. End-to-End Architecture of the Final Pipeline", fontsize=12.5, color="#223447", pad=10)

    fig.savefig(os.path.join(FIG_DIR, "RRT_ML_Paper_Illustration.png"), dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "RRT_ML_Paper_Illustration.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    # Only draw the required figures with specific names to prevent version confusion
    draw_causal_dag()
    draw_causal_dag_minimal()
    draw_method_architecture()
    print("Saved:", os.path.join(FIG_DIR, "Fig11_Causal_DAG_Generated.png"))
    print("Saved:", os.path.join(FIG_DIR, "Fig11_Causal_DAG_Generated.pdf"))
    print("Saved:", os.path.join(FIG_DIR, "RRT_ML_Paper_Illustration.png"))
    print("Saved:", os.path.join(FIG_DIR, "RRT_ML_Paper_Illustration.pdf"))


if __name__ == "__main__":
    main()

