"""Préparation des données CATMuS Medieval pour le fine-tuning Kraken.

Filtre le dataset HuggingFace ``CATMuS/medieval`` sur (langue, siècle), extrait
les lignes (image + transcription) et les écrit au format attendu par Kraken :
un couple ``line_NNNNNN.png`` + ``line_NNNNNN.gt.txt`` par ligne, organisé par
split (train / validation / test).

Stratégie réseau : les fichiers parquet (~25 Go au total, images incluses) ne
sont PAS téléchargés en entier. On utilise DuckDB avec *predicate pushdown* :
seules les colonnes ``language``/``century`` sont lues pour localiser les lignes
qui matchent, puis ``im``/``text`` ne sont récupérées que pour ces lignes-là.

Usage :
    htmir-prepare --config configs/training.yaml
    htmir-prepare --config configs/training.yaml --max-files 3 --max-samples 200
"""

import argparse
import io
import json
from pathlib import Path

import requests
import yaml

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_HF_PARQUET_API = "https://datasets-server.huggingface.co/parquet"


def get_parquet_urls(repo_id: str, split: str) -> list[str]:
    """Récupère les URLs des fichiers parquet d'un split via l'API HuggingFace.

    Args:
        repo_id: Identifiant du dataset (ex. ``"CATMuS/medieval"``).
        split: Split physique (``"train"``, ``"validation"``, ``"test"``).

    Returns:
        Liste d'URLs ``.parquet`` (vide si le split n'existe pas).
    """
    resp = requests.get(_HF_PARQUET_API, params={"dataset": repo_id}, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("parquet_files", [])
    return [f["url"] for f in files if f["split"] == split]


def _build_query(urls: list[str], language: str, century: int) -> str:
    """Construit la requête DuckDB filtrée (predicate pushdown sur language/century)."""
    url_list = ", ".join(f"'{u}'" for u in urls)
    return f"""
        SELECT im.bytes AS img_bytes, text
        FROM read_parquet([{url_list}])
        WHERE language = '{language}' AND century = {century}
          AND text IS NOT NULL AND length(trim(text)) > 0
    """


def extract_split(
    con,
    urls: list[str],
    language: str,
    century: int,
    out_dir: Path,
    start_idx: int,
    max_samples: int | None,
) -> int:
    """Extrait les lignes filtrées d'un split et les écrit sur disque.

    Args:
        con: Connexion DuckDB (httpfs chargé).
        urls: URLs parquet du split.
        language: Langue à conserver (ex. ``"French"``).
        century: Siècle à conserver (ex. ``13``).
        out_dir: Répertoire de sortie du split.
        start_idx: Index de départ pour la numérotation des lignes.
        max_samples: Nombre max de lignes à extraire (``None`` = tout).

    Returns:
        Nombre de lignes écrites.
    """
    from PIL import Image  # noqa: PLC0415 — import différé (dépendance lourde)

    out_dir.mkdir(parents=True, exist_ok=True)
    query = _build_query(urls, language, century)

    written = 0
    idx = start_idx
    # Streaming par batches (fetchmany) pour ne pas charger toutes les images en RAM.
    # Colonnes positionnelles : (img_bytes, text).
    result = con.execute(query)

    while True:
        batch = result.fetchmany(512)
        if not batch:
            break
        for img_bytes, text in batch:
            if max_samples is not None and written >= max_samples:
                return written

            text = (text or "").strip()
            if not img_bytes or not text:
                continue

            stem = f"line_{idx:06d}"
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(out_dir / f"{stem}.png", "PNG")
            except Exception as exc:
                logger.warning(f"Image invalide (idx={idx}) : {exc}")
                continue

            (out_dir / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
            written += 1
            idx += 1

            if written % 500 == 0:
                logger.info(f"  {out_dir.name} : {written} lignes écrites")

    return written


def prepare(cfg: dict, overrides: dict) -> dict:
    """Pipeline complet de préparation des données CATMuS.

    Args:
        cfg: Contenu de ``training.yaml``.
        overrides: Surcharges CLI (``max_files``, ``max_samples``, ``output_dir``).

    Returns:
        Dictionnaire récapitulatif ``{split: nb_lignes}`` + métadonnées.
    """
    import duckdb  # noqa: PLC0415

    ds = cfg["dataset"]
    repo_id = ds["hf_repo"]
    language = ds["filter"]["language"]
    century = int(ds["filter"]["century"])
    splits = ds.get("splits", ["train", "validation", "test"])

    max_files = overrides.get("max_files", ds.get("max_files"))
    max_samples = overrides.get("max_samples", ds.get("max_samples"))
    out_root = Path(overrides.get("output_dir") or cfg["output"]["local_dir"])

    logger.info(
        f"Préparation CATMuS — repo={repo_id} | filtre={language}/{century}e | "
        f"splits={splits} | max_files={max_files} | max_samples={max_samples}"
    )

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    summary: dict = {
        "repo_id": repo_id,
        "filter": {"language": language, "century": century},
        "splits": {},
        "total": 0,
    }
    global_idx = 0

    for split in splits:
        urls = get_parquet_urls(repo_id, split)
        if not urls:
            logger.warning(f"Split '{split}' : aucun fichier parquet, ignoré")
            continue
        if max_files:
            urls = urls[:max_files]

        logger.info(f"Split '{split}' : {len(urls)} fichier(s) parquet à scanner…")
        out_dir = out_root / split
        n = extract_split(
            con, urls, language, century, out_dir, global_idx, max_samples
        )
        summary["splits"][split] = n
        summary["total"] += n
        global_idx += n
        logger.info(f"Split '{split}' terminé : {n} lignes")

    # Manifeste de reproductibilité
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Manifeste écrit : {manifest_path}")
    logger.info(f"TOTAL : {summary['total']} lignes extraites dans {out_root}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prépare les données CATMuS (français médiéval) pour Kraken.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limite le nb de fichiers parquet par split (test rapide)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limite le nb de lignes extraites par split (test rapide)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Surcharge le répertoire de sortie")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    overrides = {
        "max_files": args.max_files,
        "max_samples": args.max_samples,
        "output_dir": args.output_dir,
    }
    prepare(cfg, overrides)


if __name__ == "__main__":
    main()
