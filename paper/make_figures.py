"""Génère les figures de l'article scientifique depuis les métriques réelles.

Lit ``data/catmus-french-13c/training_metrics.csv`` (entraînement) et
``data/catmus-french-13c/eval_report.json`` (bootstrap CI + IoU segmentation),
puis écrit les graphiques dans ``paper/figures/``.

Usage :
    python paper/make_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans affichage
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "data/catmus-french-13c/training_metrics.csv"
EVAL_REPORT = ROOT / "data/catmus-french-13c/eval_report.json"
OUT = ROOT / "paper/figures"
CER_TARGET = 0.08
IOU_TARGET = 0.75


def load_metrics(path: Path) -> list[dict]:
    """Charge le CSV de métriques (epoch, accuracy, cer, word_accuracy, wer)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in r.items()})
    return sorted(rows, key=lambda r: r["epoch"])


def fig_cer(rows: list[dict]) -> None:
    """Courbe du CER de validation par epoch + cible + meilleur epoch."""
    epochs = [r["epoch"] for r in rows]
    cer = [r["cer"] for r in rows]
    best_i = min(range(len(cer)), key=lambda i: cer[i])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, cer, marker="o", color="#1f4e79", label="CER validation")
    ax.axhline(CER_TARGET, ls="--", color="#c00000", label=f"Cible ({CER_TARGET:.0%})")
    ax.scatter([epochs[best_i]], [cer[best_i]], color="#2e8b57", zorder=5, s=90,
               label=f"Meilleur epoch ({epochs[best_i]:.0f}, CER {cer[best_i]:.1%})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CER")
    ax.set_title("Taux d'erreur caractère (validation) par epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cer_par_epoch.pdf")
    fig.savefig(OUT / "cer_par_epoch.png", dpi=150)
    plt.close(fig)


def fig_accuracy(rows: list[dict]) -> None:
    """Accuracy caractère vs mot par epoch."""
    epochs = [r["epoch"] for r in rows]
    char_acc = [r["accuracy"] for r in rows]
    word_acc = [r["word_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, char_acc, marker="o", color="#1f4e79", label="Accuracy caractère")
    ax.plot(epochs, word_acc, marker="s", color="#d2691e", label="Accuracy mot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Précision caractère et mot (validation) par epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_par_epoch.pdf")
    fig.savefig(OUT / "accuracy_par_epoch.png", dpi=150)
    plt.close(fig)


def fig_bootstrap_ci(report: dict) -> None:
    """Forest plot des intervalles de confiance bootstrap (CER et WER)."""
    boot = report.get("bootstrap", {})
    val = report.get("validation_best", {})
    if not boot or not boot.get("cer_ci95"):
        return

    cer, wer = val.get("cer", 0.0), val.get("wer", 0.0)
    cer_lo, cer_hi = boot["cer_ci95"]
    wer_lo, wer_hi = boot["wer_ci95"]

    fig, ax = plt.subplots(figsize=(6, 2.6))
    metrics = [
        ("WER", wer, wer_lo, wer_hi, "#d2691e"),
        ("CER", cer, cer_lo, cer_hi, "#1f4e79"),
    ]
    for y, (name, point, lo, hi, color) in enumerate(metrics):
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.1, y + 0.1], color=color, lw=2.5)
        ax.plot([hi, hi], [y - 0.1, y + 0.1], color=color, lw=2.5)
        ax.scatter([point], [y], color=color, s=110, zorder=5)
        ax.annotate(f"{point:.1%}  [{lo:.1%}, {hi:.1%}]",
                    (hi, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["WER", "CER"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, max(wer_hi * 1.45, 0.05))
    ax.set_xlabel("Taux d'erreur")
    ax.set_title(f"Intervalles de confiance bootstrap à 95% (N={boot.get('n_bootstrap', 1000)})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_ci.pdf")
    fig.savefig(OUT / "bootstrap_ci.png", dpi=150)
    plt.close(fig)


def fig_iou_segmentation(report: dict) -> None:
    """Barres IoU moyen + % lignes ≥ seuil par manuscrit d'évaluation."""
    seg = report.get("segmentation_iou", {})
    manuscripts = seg.get("manuscripts", {})
    if not manuscripts:
        return

    labels, mean_ious, pcts = [], [], []
    for ms in manuscripts.values():
        labels.append(ms.get("title", "?").split("(")[0].strip())
        mean_ious.append(ms["mean_iou"])
        pcts.append(ms["pct_above_threshold"])

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([i - width / 2 for i in x], mean_ious, width,
           color="#1f4e79", label="IoU moyen")
    ax.bar([i + width / 2 for i in x], pcts, width,
           color="#2e8b57", label=f"Lignes IoU $\\geq$ {IOU_TARGET:.0%}")
    ax.axhline(IOU_TARGET, ls="--", color="#c00000",
               label=f"Seuil ({IOU_TARGET:.0%})")

    for i, v in enumerate(mean_ious):
        ax.text(i - width / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8)
    for i, v in enumerate(pcts):
        ax.text(i + width / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Segmentation Kraken : IoU sur manuscrits HTRomance (XIIIe s.)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "iou_segmentation.pdf")
    fig.savefig(OUT / "iou_segmentation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not METRICS.exists():
        raise SystemExit(f"Métriques introuvables : {METRICS}")
    rows = load_metrics(METRICS)
    fig_cer(rows)
    fig_accuracy(rows)

    n_extra = 0
    if EVAL_REPORT.exists():
        report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
        fig_bootstrap_ci(report)
        fig_iou_segmentation(report)
        n_extra = 2
    print(f"Figures générées dans {OUT} ({len(rows)} epochs, {2 + n_extra} graphiques)")


if __name__ == "__main__":
    main()
