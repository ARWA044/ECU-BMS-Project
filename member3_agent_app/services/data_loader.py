
import json
from pathlib import Path

import pandas as pd

# Dossier principal ECU-BMS-Project
ROOT = Path(__file__).resolve().parents[2]

# Fichiers fournis par M1 et M2
CSV_PATH = (
    ROOT / "member1_simulator"
    / "datasets" / "test.csv"
)

JSON_PATH = (
    ROOT / "member2_ml"
    / "results" / "diagnoses.json"
)


def load_diagnoses():
    """Lire les diagnostics produits par M2."""

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        diagnoses = json.load(f)

    if not isinstance(diagnoses, list):
        raise ValueError(
            "diagnoses.json doit contenir une liste."
        )

    return diagnoses


def load_measurements():
    """Lire les mesures BMS produites par M1."""

    return pd.read_csv(CSV_PATH)


def get_session_data(session_id):
    """Récupérer le diagnostic et les mesures
    correspondant à une session."""

    diagnoses = load_diagnoses()
    measurements = load_measurements()

    # Sélectionner le diagnostic
    diagnosis = next(
        (
            d for d in diagnoses
            if d["session_id"] == session_id
        ),
        None
    )

    # Sélectionner les mesures
    session_df = measurements[
        measurements["scenario_id"] == session_id
    ].copy()

    return diagnosis, session_df
