"""Dashboard Streamlit — visualisation complète du projet HTmiR.

Cinq onglets : Vue d'ensemble, Dataset, Entraînement, Évaluation, Comparaison.
Toute la logique de chargement vient de :mod:`htmir.viz.data_loaders` (testée) ;
ce fichier ne fait que l'affichage.

Lancement :
    streamlit run src/htmir/viz/dashboard.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from htmir.viz import data_loaders as dl

st.set_page_config(page_title="HTmiR — Dashboard", layout="wide", page_icon="📜")

CER_TARGET = 0.08  # cible projet : CER < 8 %


# ── Sidebar : chemins des artefacts ──────────────────────────────────────────
st.sidebar.title("📜 HTmiR")
st.sidebar.caption("HTR français médiéval — XIIIe siècle (CATMuS)")

data_dir = Path(st.sidebar.text_input("Répertoire données", "data/catmus-french-13c"))
train_log = Path(st.sidebar.text_input("Log entraînement", "logs/train.log"))
metrics_csv = Path(st.sidebar.text_input("Métriques CSV", "logs/training_metrics.csv"))
eval_report = Path(st.sidebar.text_input("Rapport éval", "eval_report.json"))
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


tab_overview, tab_data, tab_train, tab_eval, tab_compare = st.tabs(
    ["Vue d'ensemble", "Dataset", "Entraînement", "Évaluation", "Comparaison"]
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

    if report:
        st.subheader("Sortie ketos test")
        st.code(report.get("ketos_output", ""), language=None)
    else:
        st.info("Pas encore de rapport d'évaluation.")

    if preds:
        from htmir.eval.evaluate import corpus_metrics
        pairs = [(p.get("ref", ""), p.get("hyp", "")) for p in preds]
        m = corpus_metrics(pairs)
        c1, c2, c3 = st.columns(3)
        c1.metric("CER (test)", f"{m['cer']:.1%}")
        c2.metric("WER (test)", f"{m['wer']:.1%}")
        c3.metric("Lignes", m["n_lines"])

        st.subheader("Pires prédictions (diagnostic)")
        for w in dl.worst_predictions(preds, n=10):
            if w.get("image_path") and Path(w["image_path"]).exists():
                st.image(w["image_path"], use_container_width=True)
            st.markdown(
                f"**CER {w['cer']:.0%}** — réf : `{w.get('ref','')}` → "
                f"préd : `{w.get('hyp','')}`"
            )
            st.divider()
    else:
        st.caption("Fournis un CSV de prédictions (ref,hyp,image_path) pour le diagnostic.")


# ── 5. COMPARAISON ───────────────────────────────────────────────────────────
with tab_compare:
    st.header("Comparaison de modèles")
    st.caption("Dépose plusieurs rapports d'éval pour comparer baseline vs fine-tuné.")
    runs_dir = Path(st.text_input("Répertoire des rapports", "runs"))
    if runs_dir.exists():
        rows = []
        for rep in sorted(runs_dir.glob("*.json")):
            data = dl.load_eval_report(rep)
            rows.append({"run": rep.stem, "output": data.get("ketos_output", "")[:80]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Aucun rapport trouvé dans ce répertoire.")
    else:
        st.info("Répertoire de comparaison inexistant pour l'instant.")
