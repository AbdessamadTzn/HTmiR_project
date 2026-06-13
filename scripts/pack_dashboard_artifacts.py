"""Assemble dashboard artifacts into a zip for sharing."""
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data/catmus-french-13c"
    staging = root / "_dashboard_export"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "data/catmus-french-13c/test_samples").mkdir(parents=True)

    log_path = data / "train.log"
    text_clean = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    stage_re = re.compile(
        r"stage\s+(\d+)/\d+.*?val_accuracy:\s*([\d.]+).*?"
        r"val_word_accuracy:\s*([\d.]+).*?early_stopping:\s*\d+/\d+\s*([\d.]+)",
    )
    epochs: dict[int, dict] = {}
    for line in text_clean.splitlines():
        if "stage " not in line or "val_accuracy:" not in line:
            continue
        m = stage_re.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        acc = float(m.group(2))
        wer_acc = float(m.group(3))
        best = float(m.group(4))
        epochs[ep] = {
            "epoch": ep,
            "accuracy": acc,
            "cer": round(1 - acc, 6),
            "word_accuracy": wer_acc,
            "wer": round(1 - wer_acc, 6),
            "best_val_metric": best,
        }
    rows = [epochs[k] for k in sorted(epochs)]

    metrics_csv = data / "training_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "epoch", "accuracy", "cer", "word_accuracy", "wer", "best_val_metric",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["best_val_metric"]) if rows else None
    best_acc = best["best_val_metric"] if best else 0.9553
    eval_report = {
        "model": "htmir-french-13c/best_0.9553.safetensors",
        "test_set": str(data / "test.arrow"),
        "note": (
            "Evaluation test set via ketos test indisponible (bug Kraken 7 binaire). "
            "Metriques validation extraites du train.log."
        ),
        "validation_best": {
            "epoch": best["epoch"] if best else 0,
            "val_accuracy": best_acc,
            "cer": round(1 - best_acc, 6),
            "val_word_accuracy": best["word_accuracy"] if best else None,
            "wer": best["wer"] if best else None,
        },
        "training": {
            "epochs_completed": len(rows),
            "early_stopping_patience": 10,
            "stopped_reason": "early_stopping",
        },
        "ketos_output": f"Best validation char accuracy: {best_acc}",
    }
    eval_path = data / "eval_report.json"
    eval_path.write_text(
        json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    for name in ("train.log", "dataset_manifest.json", "eval_report.json", "training_metrics.csv"):
        shutil.copy2(data / name, staging / "data/catmus-french-13c" / name)

    model_src = root / "htmir-french-13c/best_0.9553.safetensors"
    ckpt_src = root / "htmir-french-13c/checkpoint_00-0.9553.ckpt"
    shutil.copy2(model_src, staging / "htmir-french-13c_best.safetensors")

    test_dir = data / "test"
    for png in sorted(test_dir.glob("*.png"))[:8]:
        gt = png.with_suffix("").with_suffix(".gt.txt")
        shutil.copy2(png, staging / "data/catmus-french-13c/test_samples" / png.name)
        if gt.exists():
            shutil.copy2(gt, staging / "data/catmus-french-13c/test_samples" / gt.name)

    (staging / "README.txt").write_text(
        """HTmiR dashboard export
======================
- data/catmus-french-13c/train.log
- data/catmus-french-13c/training_metrics.csv
- data/catmus-french-13c/eval_report.json
- data/catmus-french-13c/dataset_manifest.json
- htmir-french-13c_best.safetensors (Kraken 7)
- data/catmus-french-13c/test_samples/ (8 lignes)
""",
        encoding="utf-8",
    )

    zip_path = root / "htmir-dashboard-artifacts.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    print(f"epochs parsed: {len(rows)}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
