"""Génère les figures de l'article scientifique depuis les métriques réelles.

Lit ``data/catmus-french-13c/training_metrics.csv`` (produit par l'entraînement)
et écrit les graphiques dans ``paper/figures/``.

Usage :
    python paper/make_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans affichage
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "data/catmus-french-13c/training_metrics.csv"
OUT = ROOT / "paper/figures"
CER_TARGET = 0.08


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not METRICS.exists():
        raise SystemExit(f"Métriques introuvables : {METRICS}")
    rows = load_metrics(METRICS)
    fig_cer(rows)
    fig_accuracy(rows)
    print(f"Figures générées dans {OUT} ({len(rows)} epochs)")


if __name__ == "__main__":
    main()
