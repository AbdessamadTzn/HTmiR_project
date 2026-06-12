"""Smoke test du dashboard Streamlit via AppTest.

Vérifie que l'app se charge et rend ses 5 onglets sans lever d'exception,
y compris quand aucun artefact (données, métriques, rapport) n'est présent.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = "src/htmir/viz/dashboard.py"


def test_dashboard_runs_without_data():
    """Sans données, l'app doit se charger et afficher des messages info."""
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception
    # Les 5 onglets sont présents
    assert len(at.tabs) == 5


def test_dashboard_runs_with_manifest(tmp_path, monkeypatch):
    """Avec un manifeste minimal, l'app rend la vue d'ensemble sans crash."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "dataset_manifest.json").write_text(
        '{"total": 100, "splits": {"train": 100}, '
        '"filter": {"language": "French", "century": 13}}',
        encoding="utf-8",
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    # Renseigne le répertoire de données dans la sidebar
    at.run()
    at.sidebar.text_input[0].set_value(str(data_dir)).run()
    assert not at.exception
