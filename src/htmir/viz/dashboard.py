"""Dashboard Streamlit — visualisation complète du projet HTmiR.

Cinq onglets : Vue d'ensemble, Dataset, Entraînement, Évaluation, Comparaison.
Toute la logique de chargement vient de :mod:`htmir.viz.data_loaders` (testée) ;
ce fichier ne fait que l'affichage.

Lancement :
    streamlit run src/htmir/viz/dashboard.py
"""

import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from htmir.viz import data_loaders as dl

st.set_page_config(page_title="HTmiR — Dashboard", layout="wide", page_icon="📜")

CER_TARGET = 0.08  # cible projet : CER < 8 %

_LOCAL_DATA = Path("data/catmus-french-13c")

# ── Sidebar : chemins des artefacts ──────────────────────────────────────────
st.sidebar.title("📜 HTmiR")
st.sidebar.caption("HTR français médiéval — XIIIe siècle (CATMuS)")

data_dir = Path(st.sidebar.text_input("Répertoire données", str(_LOCAL_DATA)))
train_log = Path(st.sidebar.text_input("Log entraînement", str(_LOCAL_DATA / "train.log")))
metrics_csv = Path(st.sidebar.text_input("Métriques CSV", str(_LOCAL_DATA / "training_metrics.csv")))
eval_report = Path(st.sidebar.text_input("Rapport éval", str(_LOCAL_DATA / "eval_report.json")))
preds_csv = Path(st.sidebar.text_input("Prédictions CSV", "predictions.csv"))
patience = st.sidebar.number_input("Patience (early-stopping)", 1, 50, 10)


def _load_training_rows() -> list[dict]:
    """Métriques d'entraînement : CSV prioritaire, sinon parse du log Kraken."""
    rows = dl.load_training_metrics(metrics_csv)
    if rows:
        return rows
    if train_log.exists():
        return dl.parse_ketos_log(train_log.read_text(encoding="utf-8"))
    return []


tab_overview, tab_data, tab_train, tab_eval, tab_demo, tab_nlp = st.tabs(
    ["Vue d'ensemble", "Dataset", "Entraînement", "Évaluation",
     "🔍 Tester le modèle", "🔤 NLP"]
)


# ── 1. VUE D'ENSEMBLE ────────────────────────────────────────────────────────
with tab_overview:
    st.header("Vue d'ensemble")
    manifest = dl.load_dataset_manifest(data_dir / "dataset_manifest.json")
    train_rows = _load_training_rows()
    report = dl.load_eval_report(eval_report)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes totales", manifest.get("total", "—"))
    es = dl.compute_early_stopping(train_rows, patience) if train_rows else {}
    best_cer = es.get("best_cer")
    c2.metric("Meilleur CER (val)", f"{best_cer:.1%}" if best_cer is not None else "—")
    c3.metric("Best epoch", es.get("best_epoch", "—") if es else "—")
    status = "✅ entraîné" if train_rows else "⏳ en attente"
    c4.metric("Statut", status)

    if best_cer is not None:
        st.subheader("CER vs cible (8 %)")
        pct = max(0.0, min(1.0, 1 - best_cer / CER_TARGET)) if best_cer <= CER_TARGET else 0.0
        ok = best_cer <= CER_TARGET
        st.progress(pct if ok else 0.0)
        st.caption(
            f"{'🎯 Cible atteinte' if ok else '🔴 Au-dessus de la cible'} — "
            f"CER {best_cer:.1%} (cible < {CER_TARGET:.0%})"
        )

    if manifest:
        st.json(manifest.get("filter", {}))


# ── 2. DATASET ───────────────────────────────────────────────────────────────
with tab_data:
    st.header("Dataset CATMuS — français XIIIe")
    manifest = dl.load_dataset_manifest(data_dir / "dataset_manifest.json")

    if not manifest:
        st.info("Pas encore de données préparées. Lance `htmir-prepare` d'abord.")
    else:
        splits = manifest.get("splits", {})
        df_splits = pd.DataFrame(
            {"split": list(splits.keys()), "lignes": list(splits.values())}
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Répartition")
            st.altair_chart(
                alt.Chart(df_splits).mark_bar().encode(
                    x=alt.X("split", sort=["train", "validation", "test"]),
                    y="lignes",
                    color="split",
                ),
                use_container_width=True,
            )
        with col2:
            st.subheader("Longueur des lignes (caractères)")
            all_lengths: list[int] = []
            for split in splits:
                all_lengths += dl.compute_length_stats(data_dir / split)["char_lengths"]
            if all_lengths:
                df_len = pd.DataFrame({"chars": all_lengths})
                st.altair_chart(
                    alt.Chart(df_len).mark_bar().encode(
                        x=alt.X("chars", bin=alt.Bin(maxbins=40)),
                        y="count()",
                    ),
                    use_container_width=True,
                )

        st.subheader("Caractères médiévaux spéciaux (abréviations)")
        special = {}
        for split in splits:
            for c, n in dl.char_frequency(data_dir / split, only_special=True).items():
                special[c] = special.get(c, 0) + n
        if special:
            top = dict(sorted(special.items(), key=lambda kv: -kv[1])[:20])
            df_sp = pd.DataFrame({"caractère": list(top.keys()), "freq": list(top.values())})
            st.altair_chart(
                alt.Chart(df_sp).mark_bar().encode(
                    x=alt.X("caractère", sort="-y"), y="freq",
                ),
                use_container_width=True,
            )
        else:
            st.caption("Aucun caractère spécial détecté (ou données absentes).")

        st.subheader("Exemples (image + transcription)")
        first_split = next(iter(splits), None)
        if first_split:
            for s in dl.sample_lines(data_dir / first_split, n=6):
                col_i, col_t = st.columns([3, 2])
                if Path(s["image_path"]).exists():
                    col_i.image(s["image_path"], use_container_width=True)
                col_t.code(s["text"], language=None)


# ── 3. ENTRAÎNEMENT ──────────────────────────────────────────────────────────
with tab_train:
    st.header("Courbes d'entraînement")
    train_rows = _load_training_rows()

    if not train_rows:
        st.info("Pas encore de métriques d'entraînement. Lance l'entraînement Kraken.")
    else:
        df = pd.DataFrame(train_rows)
        es = dl.compute_early_stopping(train_rows, patience)

        # Courbe CER validation + markers early-stopping
        st.subheader("CER validation par epoch")
        base = alt.Chart(df).encode(x="epoch:Q")
        line = base.mark_line(point=True).encode(y=alt.Y("cer:Q", title="CER"))
        layers = [line, base.mark_rule(color="green").encode(x="epoch:Q").transform_filter(
            alt.datum.epoch == es["best_epoch"]
        )]
        target_rule = alt.Chart(pd.DataFrame({"y": [CER_TARGET]})).mark_rule(
            color="red", strokeDash=[4, 4]
        ).encode(y="y:Q")
        st.altair_chart(alt.layer(*layers, target_rule), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Best epoch (modèle gardé)", es["best_epoch"])
        m2.metric("Best CER", f"{es['best_cer']:.1%}")
        m3.metric("Arrêt early-stopping",
                  es["stop_epoch"] if es["stop_epoch"] is not None else "non déclenché")
        st.caption(
            f"🟢 epoch {es['best_epoch']} = meilleur modèle conservé • "
            f"🟠 epochs sans amélioration : {es['stalled_epochs'] or 'aucun'} • "
            f"🔴 ligne pointillée = cible CER {CER_TARGET:.0%}"
        )

        if "accuracy" in df.columns:
            st.subheader("Accuracy caractère par epoch")
            st.altair_chart(
                alt.Chart(df).mark_line(point=True, color="orange").encode(
                    x="epoch:Q", y="accuracy:Q",
                ),
                use_container_width=True,
            )


# ── 4. ÉVALUATION ────────────────────────────────────────────────────────────
with tab_eval:
    st.header("Évaluation sur le test set")
    report = dl.load_eval_report(eval_report)
    preds = dl.load_predictions(preds_csv)

    if not report:
        st.info("Pas encore de rapport d'évaluation.")
    else:
        val = report.get("validation_best", {})
        cer = val.get("cer")
        wer = val.get("wer")

        # ── Métriques principales ─────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CER", f"{cer:.2%}" if cer is not None else "—")
        c2.metric("WER", f"{wer:.2%}" if wer is not None else "—")
        c3.metric("Epoch", val.get("epoch", "—"))
        c4.metric("Accuracy char", f"{val.get('val_accuracy', 0):.2%}" if val.get("val_accuracy") else "—")

        # ── Bootstrap CI ──────────────────────────────────────────────────
        boot = report.get("bootstrap", {})
        if boot and boot.get("cer_ci95"):
            st.subheader("Intervalles de confiance bootstrap")
            cer_lo, cer_hi = boot["cer_ci95"]
            wer_lo, wer_hi = boot["wer_ci95"]
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(f"**CER IC 95 %** : `{cer_lo:.2%}` — `{cer_hi:.2%}`")
                width_cer = cer_hi - cer_lo
                st.progress(min(1.0, cer if cer else 0.0),
                            text=f"CER {cer:.2%} (±{width_cer/2:.2%})")
            with bc2:
                st.markdown(f"**WER IC 95 %** : `{wer_lo:.2%}` — `{wer_hi:.2%}`")
                width_wer = wer_hi - wer_lo
                st.progress(min(1.0, wer if wer else 0.0),
                            text=f"WER {wer:.2%} (±{width_wer/2:.2%})")
            st.caption(
                f"Bootstrap N={boot.get('n_bootstrap', 1000)}, α={boot.get('alpha', 0.05)} "
                f"— {boot.get('note', '')}"
            )

        # ── IoU segmentation ─────────────────────────────────────────────
        iou = report.get("segmentation_iou", {})
        if iou:
            st.subheader("IoU segmentation")
            mean_iou = iou.get("mean_iou")
            pct = iou.get("pct_above_threshold")
            threshold = iou.get("threshold", 0.75)
            if mean_iou is not None:
                ic1, ic2 = st.columns(2)
                ok_iou = mean_iou >= threshold
                ic1.metric("IoU moyen (tous manuscrits)", f"{mean_iou:.2%}",
                           delta="✅ seuil atteint" if ok_iou else f"⚠️ sous seuil {threshold:.0%}")
                ic2.metric(f"Lignes IoU ≥ {threshold:.0%}",
                           f"{pct:.1%}" if pct is not None else "—")
                # Détail par manuscrit
                manuscripts = iou.get("manuscripts", {})
                if manuscripts:
                    st.caption("Détail par manuscrit (HTRomance XIIIe siècle)")
                    rows = []
                    for ms_id, ms in manuscripts.items():
                        rows.append({
                            "Manuscrit": ms.get("title", ms_id),
                            "IoU moyen": f"{ms['mean_iou']:.2%}",
                            f"≥ {threshold:.0%}": f"{ms['pct_above_threshold']:.1%}",
                            "Lignes": ms.get("n_lines", "—"),
                            "Pages": ms.get("n_pages", "—"),
                            "Seuil OK": "✅" if ms["mean_iou"] >= threshold else "⚠️",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(iou.get("note", ""))
            else:
                st.info(iou.get("note", "IoU non calculé — lancer l'évaluation sur manuscrits bruts."))

        # ── Pires prédictions ─────────────────────────────────────────────
        if preds:
            from htmir.eval.evaluate import corpus_metrics
            pairs = [(p.get("ref", ""), p.get("hyp", "")) for p in preds]
            m = corpus_metrics(pairs)
            st.subheader(f"Pires prédictions (diagnostic) — {m['n_lines']} lignes")
            for w in dl.worst_predictions(preds, n=10):
                if w.get("image_path") and Path(w["image_path"]).exists():
                    st.image(w["image_path"], use_container_width=True)
                st.markdown(
                    f"**CER {w['cer']:.0%}** — réf : `{w.get('ref','')}` → "
                    f"préd : `{w.get('hyp','')}`"
                )
                st.divider()
        else:
            st.caption("Fournis un CSV de prédictions (ref,hyp,image_path) pour le diagnostic ligne par ligne.")


# ── 6. DÉMO OCR ──────────────────────────────────────────────────────────────
with tab_demo:
    import subprocess
    import shutil
    import tempfile

    st.header("🔍 Tester le modèle sur une image")
    st.caption(
        "Upload une image de manuscrit (PNG/JPG) → Kraken segmente les lignes "
        "et transcrit avec le modèle fine-tuné."
    )

    HF_REPO = "abdessamadtouzani/htmir-french-13c"
    HF_FILE = "htmir-french-13c_best.safetensors"

    @st.cache_resource(show_spinner="Téléchargement du modèle depuis HuggingFace…")
    def _download_hf_model(repo: str, filename: str) -> str:
        """Télécharge (et met en cache) le modèle Kraken depuis HuggingFace."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo, filename=filename)

    st.caption(f"Modèle : `{HF_REPO}` (chargé directement depuis HuggingFace)")

    uploaded = st.file_uploader("Image manuscrit", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img_path = tmp / uploaded.name
            img_path.write_bytes(uploaded.read())

            col_img, col_txt = st.columns([1, 1])
            col_img.subheader("Image")
            col_img.image(str(img_path), use_container_width=True)

            # Modèle : téléchargé (et mis en cache) directement depuis HuggingFace.
            model_path = None
            model_error = None
            try:
                model_path = Path(_download_hf_model(HF_REPO, HF_FILE))
            except Exception as exc:  # noqa: BLE001
                model_error = str(exc)

            # Priorité au kraken HTR du venv (évite le kraken bio-informatique apt)
            venv_kraken = Path(__file__).resolve().parents[3] / ".venv/bin/kraken"
            kraken_bin = str(venv_kraken) if venv_kraken.exists() else shutil.which("kraken")
            if not kraken_bin or not Path(kraken_bin).exists():
                col_txt.error(
                    "Kraken introuvable. Installe-le avec :\n```\npip install kraken\n```"
                )
            elif model_path is None or not model_path.exists():
                col_txt.error(
                    "Modèle indisponible.\n\n"
                    f"Échec du chargement depuis `{HF_REPO}` "
                    f"({model_error or 'fichier introuvable'})."
                )
            else:
                # Sortie ALTO XML (coordonnées + texte) — format du corpus HTRomance.
                out_xml = tmp / "output.xml"
                cmd = [
                    kraken_bin, "-a",  # -a : sérialisation ALTO XML
                    "-i", str(img_path), str(out_xml),
                    "segment", "-bl",
                    "ocr", "-m", str(model_path.resolve()),
                ]
                col_txt.subheader("Transcription")
                with col_txt:
                    with st.spinner("Segmentation + reconnaissance en cours…"):
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=300
                        )
                    if result.returncode != 0:
                        st.error(f"Erreur Kraken :\n```\n{result.stderr[-1000:]}\n```")
                    elif out_xml.exists():
                        alto = out_xml.read_text(encoding="utf-8")
                        texte = dl.text_from_alto(alto)
                        st.text_area("Résultat", texte, height=350)
                        lignes = [l for l in texte.splitlines() if l.strip()]
                        st.caption(f"{len(lignes)} ligne(s) transcrite(s)")
                        dc1, dc2 = st.columns(2)
                        dc1.download_button(
                            "⬇️ ALTO XML",
                            data=alto,
                            file_name=f"{img_path.stem}.xml",
                            mime="application/xml",
                        )
                        dc2.download_button(
                            "⬇️ Texte brut",
                            data=texte,
                            file_name=f"{img_path.stem}.txt",
                            mime="text/plain",
                        )
                    else:
                        st.warning("Kraken n'a produit aucune sortie.")
    else:
        st.info("Upload une image ci-dessus pour lancer la transcription.")


# ── 7. NLP (post-traitement, lecture Supabase) ───────────────────────────────
with tab_nlp:
    import os

    st.header("🔤 NLP — post-traitement & CER avant/après")
    st.caption(
        "Runs d'évaluation (data contract → normalisation → CER) stockés sur "
        "Supabase. Voir `CONVENTIONS_NLP.md` pour la méthodologie."
    )

    def _database_url() -> str | None:
        # 1. Secrets Streamlit (prod déployée)
        try:
            if "DATABASE_URL" in st.secrets:
                return st.secrets["DATABASE_URL"]
        except Exception:  # noqa: BLE001
            pass
        # 2. Variable d'environnement
        if os.environ.get("DATABASE_URL"):
            return os.environ["DATABASE_URL"]
        # 3. Fichier .env à la racine (dev local, non versionné)
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DATABASE_URL="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    @st.cache_data(ttl=60, show_spinner="Lecture des runs Supabase…")
    def _load_runs(url: str) -> list[dict]:
        from htmir.nlp.db import fetch_runs
        return fetch_runs(database_url=url)

    db_url = _database_url()
    if not db_url:
        st.info(
            "Base non configurée. Ajoute `DATABASE_URL` dans les *Secrets* "
            "Streamlit (ou en variable d'environnement) pour afficher les runs."
        )
    else:
        try:
            runs = _load_runs(db_url)
        except Exception as exc:  # noqa: BLE001
            runs = []
            st.error(f"Connexion Supabase échouée : {exc}")

        if not runs:
            st.warning("Aucun run enregistré pour l'instant.")
        else:
            df = pd.DataFrame(runs)
            latest = df.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CER brut", f"{latest['cer_raw']:.2%}"
                      if latest.get("cer_raw") is not None else "—")
            c2.metric("CER normalisé", f"{latest['cer_normalized']:.2%}"
                      if latest.get("cer_normalized") is not None else "—")
            delta = (latest["cer_raw"] - latest["cer_normalized"]) \
                if latest.get("cer_raw") and latest.get("cer_normalized") else None
            c3.metric("Gain normalisation",
                      f"{delta:+.2%}" if delta is not None else "—")
            c4.metric("Taux needs_review", f"{latest['needs_review_rate']:.1%}"
                      if latest.get("needs_review_rate") is not None else "—")

            # CER brut vs normalisé par run
            st.subheader("CER avant / après par run")
            plot_df = df[["manuscript", "cer_raw", "cer_normalized"]].melt(
                id_vars="manuscript", var_name="mesure", value_name="cer"
            )
            st.altair_chart(
                alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("manuscript:N", title="Manuscrit"),
                    xOffset="mesure:N",
                    y=alt.Y("cer:Q", title="CER", axis=alt.Axis(format="%")),
                    color="mesure:N",
                    tooltip=["manuscript", "mesure", alt.Tooltip("cer", format=".2%")],
                ),
                use_container_width=True,
            )

            st.subheader("Détail des runs")
            st.dataframe(
                df[["created_at", "manuscript", "title", "n_pages", "n_lines",
                    "cer_raw", "cer_normalized", "needs_review_rate", "iou_mean"]],
                use_container_width=True, hide_index=True,
            )


