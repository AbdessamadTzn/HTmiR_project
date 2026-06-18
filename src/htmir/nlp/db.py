"""Accès Supabase (Postgres) pour stocker et lire les runs NLP.

Connexion via ``DATABASE_URL`` (chaîne Postgres Supabase). Deux tables :
``runs`` (un run = un manuscrit évalué) et ``lines`` (détail par ligne).

Lecture seule côté dashboard ; écriture côté pipeline.
"""

import os

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - dépendance optionnelle
    psycopg2 = None


def _connect(database_url: str | None = None):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 non installé (pip install psycopg2-binary).")
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL absent (env ou secrets Streamlit).")
    return psycopg2.connect(url)


def insert_run(run: dict, lines: list[dict] | None = None,
               database_url: str | None = None) -> int:
    """Insère un run (+ ses lignes) et retourne l'``id`` du run.

    Args:
        run: Champs de la table ``runs`` (manuscript, title, model, cer_*, …).
        lines: Lignes optionnelles (table ``lines``).
        database_url: Override de la connexion.
    """
    conn = _connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cols = ["manuscript", "title", "model", "n_pages", "n_lines",
                    "cer_raw", "cer_normalized", "wer_raw", "wer_normalized",
                    "iou_mean", "needs_review_rate", "notes"]
            vals = [run.get(c) for c in cols]
            placeholders = ", ".join(["%s"] * len(cols))
            cur.execute(
                f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders}) "
                "RETURNING id",
                vals,
            )
            run_id = cur.fetchone()[0]

            if lines:
                rows = [
                    (run_id, ln.get("line_index"), ln.get("text_raw"),
                     ln.get("text_normalized"), ln.get("text_gt"),
                     ln.get("mean_confidence"), ln.get("needs_review"),
                     ln.get("cer_line"))
                    for ln in lines
                ]
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO lines (run_id, line_index, text_raw, "
                    "text_normalized, text_gt, mean_confidence, needs_review, "
                    "cer_line) VALUES %s",
                    rows,
                )
        return run_id
    finally:
        conn.close()


def fetch_runs(limit: int = 100, database_url: str | None = None) -> list[dict]:
    """Récupère les derniers runs (pour le dashboard)."""
    conn = _connect(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def fetch_lines(run_id: int, database_url: str | None = None) -> list[dict]:
    """Récupère les lignes d'un run."""
    conn = _connect(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM lines WHERE run_id = %s ORDER BY line_index",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
