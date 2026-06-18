"""Volet NLP de HTmiR : post-traitement des transcriptions HTR.

Pipeline : ALTO XML (sortie Kraken) → data contract JSON → normalisation par
règles → mesure du CER avant/après. L'entrée obligatoire de tout le NLP est le
**data contract** (cf. :mod:`htmir.nlp.data_contract`).
"""
