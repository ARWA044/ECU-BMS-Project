"""ECU / BMS Intelligence — interface M3 intégrée au dépôt commun.

Lancer depuis la racine de ECU-BMS-Project :
    python -m streamlit run member3_agent_app/app.py

Sources :
- member1_simulator/datasets/test.csv (signaux synthétiques par session)
- member2_ml/results/diagnoses.json (sorties du pipeline de M2)

Cette interface NE lance ni entraînement, ni modèle ML, ni LLM.
Elle n'invente aucun verdict fonctionnel PASS/FAIL.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from member3_agent_app.services.data_loader import load_diagnoses, load_measurements

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "member1_simulator" / "datasets" / "test.csv"
REPORTS_PATH = ROOT / "member2_ml" / "results" / "diagnoses.json"
RULES_PATH = ROOT / "member2_ml" / "config" / "rules.json"

SIGNALS = {
    "pack_voltage_v": ("Tension de la batterie", "V"),
    "pack_current_a": ("Courant de la batterie", "A"),
    "temperature_c": ("Température de la batterie", "°C"),
    "soc_pct": ("État de charge", "%"),
}

st.set_page_config(
    page_title="BMS Intelligence | Diagnostic Studio",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
  :root {
    --bg:#0c141a;
    --sidebar:#101b23;
    --panel:#17232d;
    --panel-hover:#1c2a35;
    --line:#2b3b47;
    --accent:#8ed9bb;
    --accent-soft:rgba(142,217,187,.10);
    --ink:#f0f4f4;
    --muted:#a7b8c2;
    --subtle:#7f95a3;
  }
  html, body, .stApp, [data-testid="stSidebar"],
  [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] *,
  button, input, textarea {
    font-family:'DM Sans', 'Segoe UI Variable', 'Segoe UI', sans-serif !important;
  }
  .stApp {
    background:
      radial-gradient(ellipse 500px 400px at 95% -5%, rgba(108,151,133,.075), transparent 72%),
      var(--bg);
    color:var(--ink);
  }
  [data-testid="stSidebar"] {
    background:var(--sidebar) !important;
    border-right:1px solid var(--line);
  }
  [data-testid="stHeader"] { background:transparent; }
  .block-container { padding-top:2.45rem; padding-bottom:3rem; max-width:1450px; }
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3 {
    font-family:'Manrope','Segoe UI Variable',sans-serif !important;
    letter-spacing:-.035em;
    color:var(--ink);
  }
  [data-testid="stMarkdownContainer"] h3 {font-size:1.18rem; font-weight:700;}
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {line-height:1.66;}
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    font-size:.9rem;
  }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:var(--subtle)!important; font-size:.76rem; }
  div[data-baseweb="select"] > div {
    background:var(--panel)!important;
    border:1px solid var(--line)!important;
    color:var(--ink)!important;
    border-radius:10px!important;
  }
  [data-testid="stFileUploader"] {
    border:1px dashed #3c505e;
    border-radius:12px;
    background:rgba(155,190,210,.035);
  }
  [data-testid="stFileUploader"] section { background:transparent; }
  [data-testid="stTabs"] [role="tablist"] {
    gap:1.2rem; border-bottom:1px solid var(--line); padding-bottom:2px;
  }
  [data-testid="stTabs"] button[role="tab"] {
    border-radius:0; padding:.75rem .2rem; color:var(--muted);
    font-size:.88rem; font-weight:600;
  }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:var(--accent)!important;
  }
  [data-testid="stDataFrame"] {
    border:1px solid var(--line); border-radius:12px; overflow:hidden;
  }
  [data-testid="stMetricLabel"] {font-size:.8rem!important;color:var(--muted)!important;}
  [data-testid="stMetricValue"] {font-family:'Manrope',sans-serif!important;color:var(--ink)!important;}
  [data-testid="stAlert"] {border-radius:12px;}
  .brand-wrap { display:flex;gap:12px;align-items:center;margin:.3rem 0 1.2rem; }
  .brand-mark {width:43px;height:43px;border-radius:13px;background:var(--accent-soft);
               border:1px solid rgba(158,215,188,.23);display:flex;align-items:center;
               justify-content:center;color:var(--accent);font-family:'Manrope',sans-serif;
               font-size:1.19rem;font-weight:800;}
  .brand { color:var(--ink);font-family:'Manrope',sans-serif;font-size:1.1rem;font-weight:800;
           letter-spacing:-.03em;line-height:1.25; }
  .brand-sub {color:var(--subtle);font-size:.76rem;font-weight:500;margin-top:3px;}
  .sidebar-section {color:var(--muted);font-size:.78rem;font-weight:700;
                    letter-spacing:.015em;margin:1rem 0 .35rem;}
  .hero {
    position:relative;overflow:hidden;border:1px solid var(--line);
    border-radius:20px;padding:34px 38px 30px;margin-bottom:24px;
    background:linear-gradient(115deg,rgba(29,47,55,.94),rgba(21,34,43,.97) 68%,rgba(26,39,43,.96));
  }
  .hero:after { content:'';position:absolute;right:-55px;top:-125px;width:340px;height:340px;
                border:1px solid rgba(157,203,181,.11);border-radius:50%;
                box-shadow:0 0 0 64px rgba(148,204,175,.022),0 0 0 130px rgba(148,204,175,.012);
                pointer-events:none; }
  .eyebrow {font-size:.74rem;letter-spacing:.075em;font-weight:700;color:var(--accent);}
  .hero h1 {position:relative;z-index:1;
            font-family:'Manrope','Segoe UI Variable',sans-serif!important;
            font-size:clamp(1.86rem,3vw,2.8rem); letter-spacing:-.055em;
            margin:.75rem 0 .85rem;font-weight:750;line-height:1.22;color:var(--ink);}
  .hero h1 em {font-style:normal;color:var(--accent);}
  .hero p {position:relative;z-index:1;max-width:680px;color:var(--muted);
           font-size:.96rem;line-height:1.72;margin:0;}
  .hero-tags {margin-top:24px;display:flex;gap:8px;flex-wrap:wrap;}
  .tag {display:inline-block;border:1px solid #40525a;border-radius:8px;padding:6px 10px;
        background:rgba(195,216,209,.035);color:#bfced3;font-size:.74rem;font-weight:600;}
  .section-label {color:var(--muted);font-size:.87rem;letter-spacing:.01em;
                  font-weight:700;margin:1.35rem 0 .75rem;}
  .metric-card {min-height:131px;border:1px solid var(--line);border-radius:14px;
                padding:19px 21px;background:var(--panel);}
  .metric-label {font-size:.81rem;color:var(--muted);font-weight:600;}
  .metric-value {font-family:'Manrope',sans-serif!important;font-size:clamp(1.1rem,1.9vw,1.49rem);
                 letter-spacing:-.035em;line-height:1.35;font-weight:750;
                 color:var(--ink);margin:13px 0 6px;overflow-wrap:anywhere;}
  .metric-detail {font-size:.73rem;color:var(--subtle);line-height:1.45;}
  .tone-alert {color:#f1a397!important;} .tone-good {color:#9ee1bc!important;}
  .tone-cyan {color:var(--accent)!important;} .tone-violet{color:#d3c4ef!important;}
  .info-card {min-height:155px;background:var(--panel);border:1px solid var(--line);
              border-radius:14px;padding:23px 25px;margin:5px 0 14px;}
  .info-title {color:var(--muted);font-size:.81rem;font-weight:650;margin-bottom:13px;}
  .info-big {font-family:'Manrope',sans-serif!important;color:var(--ink);font-size:1.26rem;
             letter-spacing:-.03em;line-height:1.46;font-weight:730;overflow-wrap:anywhere;margin-bottom:9px;}
  .info-body {color:var(--muted);font-size:.87rem;line-height:1.7;}
  .event {display:flex;gap:17px;padding:15px 0;border-bottom:1px solid var(--line);}
  .event-time {min-width:68px;color:var(--accent);font-weight:750;
               font-variant-numeric:tabular-nums;font-size:.84rem;}
  .event-label {font-weight:700;color:var(--ink);font-size:.9rem;overflow-wrap:anywhere;}
  .event-desc {font-size:.82rem;color:var(--muted);margin-top:4px;line-height:1.58;}
  .notice {background:rgba(227,181,112,.065);border:1px solid rgba(222,184,122,.24);
           border-radius:12px;color:#e5caa6;padding:13px 17px;font-size:.86rem;
           line-height:1.6;margin:9px 0 19px;}
  .footer {border-top:1px solid var(--line);padding-top:18px;margin-top:35px;
           color:var(--subtle);font-size:.8rem;line-height:1.6;}
  @media(max-width:800px){
    .hero{padding:24px 22px;}.hero:after{display:none;}
    .metric-card{min-height:110px;padding:16px;}
    .metric-value{font-size:1.09rem;}
    .info-card{padding:19px;}
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def safe(value: Any) -> str:
    """Échappe les valeurs placées dans les panneaux HTML stylisés."""
    return html.escape("—" if value is None else str(value), quote=True)


def norm_number(value: Any) -> float | None:
    try:
        n = float(value)
        return n if pd.notna(n) and abs(n) != float("inf") else None
    except (TypeError, ValueError):
        return None


FAULT_LABELS = {
    "NORMAL": "Aucun défaut identifié",
    "UNDERVOLTAGE": "Sous-tension",
    "OVERVOLTAGE": "Surtension",
    "OVERCURRENT": "Surintensité",
    "OVERTEMPERATURE": "Surchauffe",
    "VOLTAGE_SENSOR_FAULT": "Capteur de tension",
    "CURRENT_SENSOR_FAULT": "Capteur de courant",
    "TEMPERATURE_SENSOR_FAULT": "Capteur thermique",
}


def readable_fault(value: Any) -> str:
    if not value:
        return "Aucun défaut retenu"
    return FAULT_LABELS.get(str(value), str(value).replace("_", " ").capitalize())


def readable_status(value: Any) -> str:
    values = {
        "suspected": "À confirmer",
        "uncertain": "Incertain",
        "no_anomaly_detected": "Aucune anomalie détectée",
        "confirmed": "Confirmé",
    }
    return values.get(str(value), str(value).replace("_", " ").capitalize())


def metric(label: str, value: Any, sub: str, tone: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{safe(label)}</div>'
        f'<div class="metric-value {safe(tone)}">{safe(value)}</div>'
        f'<div class="metric-detail">{safe(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def card(title: str, headline: Any, description: str = "") -> None:
    st.markdown(
        f'<div class="info-card"><div class="info-title">{safe(title)}</div>'
        f'<div class="info-big">{safe(headline)}</div>'
        f'<div class="info-body">{safe(description)}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def default_reports() -> list[dict]:
    """Lit les résultats déjà produits par M2 ; ne lance pas le modèle."""
    return load_diagnoses()


@st.cache_data(show_spinner=False)
def default_measurements() -> pd.DataFrame:
    """Lit les mesures simulées par M1 ; pas les étiquettes de vérité terrain."""
    return load_measurements()


@st.cache_data(show_spinner=False)
def get_rules() -> list[dict]:
    """Seuils illustratifs de M2, affichés mais jamais présentés comme exigences OEM."""
    if not RULES_PATH.is_file():
        return []
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return raw.get("rules", []) if isinstance(raw, dict) else []


def validate_reports(value: Any) -> list[dict]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("Le fichier de diagnostic doit être une liste JSON ou un objet JSON.")
    if not value or not all(isinstance(d, dict) and d.get("session_id") for d in value):
        raise ValueError("Chaque diagnostic doit être un objet avec un champ session_id.")
    identifiers = [str(d["session_id"]) for d in value]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Plusieurs diagnostics portent le même session_id.")
    return value


def validate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario_id", "timestamp_s", *SIGNALS.keys()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Colonnes CSV manquantes : " + ", ".join(missing))
    data = df[["scenario_id", "timestamp_s", *SIGNALS.keys()]].copy()
    data["scenario_id"] = data["scenario_id"].astype(str)
    for name in ("timestamp_s", *SIGNALS.keys()):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["timestamp_s"])
    return data


def format_score(cause: dict) -> str:
    score = cause.get("score")
    return "indisponible" if score is None else f"{score} (score du modèle, non calibré)"


def make_report(d: dict) -> str:
    """Rapport purement descriptif : n'ajoute aucune affirmation causale."""
    lines = [
        "# Rapport de diagnostic BMS — données synthétiques", "",
        f"**Session :** {d.get('session_id','?')}",
        f"**Verdict fonctionnel :** {d.get('verdict','NOT_EVALUATED')}",
        f"**Limites de mesure :** {d.get('measurement_limits_status','UNKNOWN')}",
        f"**Défaut suspecté :** {d.get('detected_fault') or 'Aucun défaut retenu'}",
        f"**Statut :** {d.get('diagnostic_status','unknown')}",
        f"**Source :** {d.get('data_source','non précisée')}",
        f"**Modèle :** {d.get('model_version','non précisé')}",
        f"**Règles :** {d.get('ruleset_version','non précisées')}",
        "", "## Hypothèses du modèle",
    ]
    for cause in d.get("probable_causes", []):
        lines.append(
            f"- {cause.get('code','?')} — {cause.get('method','?')} : "
            f"{format_score(cause)}; interprétation : "
            f"{cause.get('score_interpretation','non fournie')}"
        )
    if not d.get("probable_causes"):
        lines.append("- Aucune cause proposée.")
    anomaly = d.get("anomaly_detection") or {}
    lines += [
        "", "## Détection d'anomalies",
        f"- Méthode : {anomaly.get('method','non fournie')}",
        f"- Anomalie détectée : {anomaly.get('anomaly_detected','unknown')}",
        f"- Fenêtres anormales : {anomaly.get('anomalous_windows','?')} / "
        f"{anomaly.get('total_windows','?')}",
        "", "## Preuves mesurées",
    ]
    for ev in d.get("evidence", []):
        lines.append(
            f"- t={ev.get('timestamp_s','?')} s : {ev.get('signal','?')} = "
            f"{ev.get('value','?')} {ev.get('unit','')} — "
            f"{ev.get('supports','non précisé')}"
        )
    if not d.get("evidence"):
        lines.append("- Aucune preuve enregistrée.")
    lines += ["", "## Vérification recommandée", str(d.get("recommended_test") or "Non renseignée")]
    lines += [
        f"Test réalisé : {'oui' if d.get('recommended_test_done') is True else 'non / non confirmé'}",
        "", "## Informations manquantes",
    ]
    lines.extend(f"- {item}" for item in d.get("missing_info", []))
    if not d.get("missing_info"):
        lines.append("- Aucun manque renseigné (ne prouve pas l'exhaustivité).")
    lines += [
        "", "---", "Les valeurs proviennent d'une simulation BMS. "
        "Les règles sont illustratives et aucun PASS/FAIL fonctionnel n'a été évalué. "
        "Les hypothèses ne sont pas des causes physiques confirmées. "
        "Ce rapport est un assemblage descriptif sans LLM ni test exécuté."
    ]
    return "\n".join(lines)


# --------------------------- Chargement et sélection ---------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-wrap"><div class="brand-mark">B</div>'
        '<div><div class="brand">BMS Studio</div>'
        '<div class="brand-sub">Analyse & diagnostic intelligent</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Source des diagnostics**")
    uploaded_reports = st.file_uploader("JSON de M2 (facultatif)", type=["json"])
    st.caption("Par défaut : member2_ml/results/diagnoses.json")
    st.markdown("**Source des mesures**")
    uploaded_measurements = st.file_uploader("CSV de M1 (facultatif)", type=["csv"])
    st.caption("Par défaut : member1_simulator/datasets/test.csv")

try:
    if uploaded_reports is None:
        reports = validate_reports(default_reports())
    else:
        reports = validate_reports(json.loads(uploaded_reports.getvalue().decode("utf-8-sig")))
    if uploaded_measurements is None:
        measurements = validate_measurements(default_measurements())
    else:
        measurements = validate_measurements(pd.read_csv(uploaded_measurements))
    rules = get_rules()
except (OSError, ValueError, UnicodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"Impossible de charger les données M1 / M2 : {exc}")
    st.caption("Vérifie les chemins des fichiers dans le dépôt commun et la structure des colonnes.")
    st.stop()

by_id = {str(item["session_id"]): item for item in reports}
session_ids = sorted(by_id)
default_index = session_ids.index("F02_R002") if "F02_R002" in by_id else 0

with st.sidebar:
    st.markdown("**Session à analyser**")
    session_id = st.selectbox(
        "Choisir une session BMS", session_ids, index=default_index,
        format_func=lambda s: f"{s} · {by_id[s].get('detected_fault') or 'aucun défaut identifié'}",
    )
    st.caption(f"{len(session_ids)} rapports disponibles · {len(measurements):,} lignes BMS".replace(",", " "))
    st.divider()
    st.caption("Données synthétiques · M1 → M2 → M3")
    st.caption("Analyse du ML déjà calculée · Module agentique en préparation.")

report = by_id[session_id]
samples = measurements[measurements["scenario_id"] == session_id].sort_values("timestamp_s").copy()
evidence = [x for x in report.get("evidence", []) if isinstance(x, dict)]
causes = [x for x in report.get("probable_causes", []) if isinstance(x, dict)]
anomaly = report.get("anomaly_detection") or {}
verdict = str(report.get("verdict") or "NOT_EVALUATED").upper()
limits = str(report.get("measurement_limits_status") or "UNKNOWN")
fault = str(report.get("detected_fault") or "Aucun défaut identifié")
status = str(report.get("diagnostic_status") or "unknown")

st.markdown(
    '<div class="hero"><div class="eyebrow">BMS Studio · Analyse des batteries</div>'
    '<h1>Comprendre les signaux.<br/><em>Expliquer les anomalies.</em></h1>'
    '<p>Explorez les mesures de la batterie, les prédictions du modèle et '
    'les éléments de preuve — dans un même espace de diagnostic.</p>'
    '<div class="hero-tags"><span class="tag">Données synthétiques</span>'
    '<span class="tag">Random Forest</span><span class="tag">Isolation Forest</span>'
    '<span class="tag">Rapports traçables</span></div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice"><b>À savoir.</b> Prototype académique basé sur des mesures simulées et '
    'seuils illustratifs. Le diagnostic ML n’est pas une validation ECU industrielle. '
    'Le verdict fonctionnel PASS/FAIL n’a pas été évalué.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Session en cours</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    metric("Session", session_id, "Identifiant de simulation", "tone-cyan")
with col2:
    metric("Verdict fonctionnel", "Non évalué" if verdict == "NOT_EVALUATED" else verdict, f"Code : {verdict}", "tone-alert" if verdict == "FAIL" else "")
with col3:
    metric("Limites de mesure", "Seuil dépassé" if limits == "LIMIT_VIOLATION" else "Dans les limites", f"Code : {limits}", "tone-alert" if limits == "LIMIT_VIOLATION" else "tone-good")
with col4:
    metric("Défaut suspecté", readable_fault(report.get("detected_fault")), f"Code : {fault}", "tone-violet")

if samples.empty:
    st.warning(
        f"Aucune mesure BMS associée à {session_id} dans le CSV chargé. "
        "Les diagnostics restent disponibles, mais les courbes ne peuvent pas être affichées."
    )

st.markdown('<div class="section-label">Explorer le diagnostic</div>', unsafe_allow_html=True)
t_overview, t_signals, t_ml, t_report = st.tabs([
    "Vue d’ensemble", "Signaux & preuves", "Analyse ML", "Rapport"
])

with t_overview:
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        card("Diagnostic du moteur ML", readable_fault(report.get("detected_fault")),
             "Hypothèse établie par le moteur M2 à partir des mesures synthétiques. "
             "Elle ne confirme pas une cause physique.")
        st.markdown("**Chronologie des observations**")
        for ev in sorted(evidence, key=lambda item: norm_number(item.get("timestamp_s")) or 0):
            st.markdown(
                f'<div class="event"><div class="event-time">{safe(ev.get("timestamp_s", "?"))} s</div>'
                f'<div><div class="event-label">{safe(ev.get("signal", "?"))}</div>'
                f'<div class="event-desc">Valeur : {safe(ev.get("value", "?"))} '
                f'{safe(ev.get("unit", ""))} · {safe(ev.get("supports", ""))}'
                '</div></div></div>', unsafe_allow_html=True,
            )
        if not evidence:
            st.info("Aucune preuve horodatée enregistrée pour cette session.")
    with right:
        card("Statut de l’analyse", readable_status(status),
             f"Détection d'anomalie : {anomaly.get('anomaly_detected','non indiquée')} · "
             f"{anomaly.get('anomalous_windows',0)} fenêtres sur {anomaly.get('total_windows',0)}")
        top = causes[0] if causes else {}
        card("Hypothèse prioritaire", readable_fault(top.get("code")) if top else "Aucune cause proposée",
             f"Méthode : {top.get('method','non renseignée')} · "
             f"Score : {format_score(top) if top else 'non disponible'}")
        card("Vérification recommandée", report.get("recommended_test") or "Non fournie",
             "Non réalisée / non confirmée" if not report.get("recommended_test_done") else "Signalée comme réalisée")

with t_signals:
    st.subheader("Évolution des signaux BMS")
    st.caption("Les courbes proviennent exclusivement de la session sélectionnée dans test.csv.")
    if not samples.empty:
        available = [col for col in SIGNALS if col in samples.columns and samples[col].notna().any()]
        if available:
            evidence_signals = [ev.get("signal") for ev in evidence]
            default_signal = next((k for k in evidence_signals if k in available), available[0])
            signal = st.selectbox(
                "Signal à visualiser", available, index=available.index(default_signal),
                format_func=lambda k: f"{SIGNALS[k][0]} ({SIGNALS[k][1]})",
            )
            plot_data = samples[["timestamp_s", signal]].dropna().copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data["timestamp_s"], y=plot_data[signal], mode="lines",
                name=SIGNALS[signal][0], line=dict(color="#8ed9bb", width=3),
                hovertemplate="t=%{x:.1f} s<br>Valeur=%{y:.3f}<extra></extra>",
            ))
            # Fenêtres anormales signalées par le moteur Isolation Forest.
            for i, window in enumerate(anomaly.get("windows", [])):
                x0 = norm_number(window.get("start_s"))
                x1 = norm_number(window.get("end_s"))
                if x0 is not None and x1 is not None and x1 > x0:
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="#bcb0d6", opacity=0.10,
                                  line_width=0, layer="below")
            # Seuils configurés par M2 pour le signal, pas des exigences OEM.
            for rule in rules:
                if rule.get("signal") == signal:
                    limit = norm_number(rule.get("limit"))
                    if limit is not None:
                        fig.add_hline(y=limit, line_dash="dash", line_color="#e8b37e",
                                      annotation_text=f"{rule.get('id','seuil')} ({rule.get('op','')}{limit:g})",
                                      annotation_font_color="#e8b37e")
            # Marqueurs provenant du JSON M2.
            for ev in evidence:
                if ev.get("signal") != signal:
                    continue
                x, y = norm_number(ev.get("timestamp_s")), norm_number(ev.get("value"))
                if x is not None and y is not None:
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y], mode="markers", name="Preuve M2",
                        marker=dict(size=12, color="#f1be82", symbol="diamond",
                                    line=dict(width=1.5, color="#17232d")),
                        hovertemplate="Preuve : t=%{x:.1f} s<br>Valeur=%{y:.3f}<extra></extra>",
                    ))
            fig.update_layout(
                height=450, margin=dict(l=20, r=20, t=35, b=20),
                paper_bgcolor="#17232d", plot_bgcolor="#17232d",
                font=dict(family="DM Sans, Segoe UI, sans-serif", color="#c9d4d8", size=13),
                xaxis=dict(title="Temps (s)", gridcolor="rgba(160,185,190,.12)", zeroline=False,
                           showline=True, linecolor="#364751"),
                yaxis=dict(title=f"{SIGNALS[signal][0]} ({SIGNALS[signal][1]})",
                           gridcolor="rgba(160,185,190,.12)", zeroline=False),
                legend=dict(orientation="h", y=1.18, x=0), hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Bande violette = fenêtre anormale du modèle ; losange orange = preuve M2 ; "
                "ligne orange pointillée = limite configurée (illustrative)."
            )
            a, b, c = st.columns(3)
            a.metric("Valeur minimale", f"{plot_data[signal].min():.2f} {SIGNALS[signal][1]}")
            b.metric("Valeur maximale", f"{plot_data[signal].max():.2f} {SIGNALS[signal][1]}")
            c.metric("Échantillons", len(plot_data))
        st.markdown("**Données mesurées (sans étiquettes de simulation)**")
        st.dataframe(samples[["timestamp_s", *SIGNALS.keys()]], use_container_width=True,
                     hide_index=True, height=265)
    st.markdown("**Preuves citées par M2**")
    if evidence:
        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
    else:
        st.info("Aucune preuve ajoutée par le moteur ML pour cette session.")

with t_ml:
    a, b = st.columns([1.12, .88], gap="large")
    with a:
        st.subheader("Détection & classification")
        if causes:
            for i, cause in enumerate(causes, start=1):
                card(f"Hypothèse {i}", cause.get("code") or "Non renseignée",
                     f"Modèle : {cause.get('method','non précisé')} · "
                     f"Score : {format_score(cause)} · "
                     f"Interprétation : {cause.get('score_interpretation','non précisée')}")
        else:
            st.info("Aucune cause probable fournie par le classifieur.")
        x, y, z = st.columns(3)
        x.metric("Anomalie IF", "Oui" if anomaly.get("anomaly_detected") is True else "Non")
        y.metric("Fenêtres atypiques", anomaly.get("anomalous_windows", 0))
        z.metric("Fenêtres totales", anomaly.get("total_windows", 0))
        if anomaly.get("windows"):
            st.dataframe(pd.DataFrame(anomaly["windows"]), hide_index=True,
                         use_container_width=True)
    with b:
        st.subheader("Vérifications proposées")
        card("Contrôle recommandé", report.get("recommended_test") or "Aucun",
             "Ce tableau de bord ne réalise aucune action sur un ECU.")
        if report.get("recommended_test_done") is True:
            st.success("Test déclaré effectué dans le diagnostic de M2.")
        else:
            st.warning("Le test recommandé n'a pas été effectué ou confirmé.")
        st.markdown("**Informations manquantes**")
        for item in report.get("missing_info", []):
            st.write("•", str(item))
        if not report.get("missing_info"):
            st.info("Aucune information manquante explicitement répertoriée.")
        st.markdown("**Versions**")
        st.caption(f"Modèle : {report.get('model_version','?')} · Règles : {report.get('ruleset_version','?')}")

with t_report:
    st.subheader("Rapport de la session")
    st.caption("Reprend exclusivement les résultats de M2. Aucun LLM ni vérification supplémentaire.")
    md = make_report(report)
    st.markdown(md)
    st.download_button(
        "⬇ Télécharger le rapport Markdown", data=md.encode("utf-8"),
        file_name=f"BMS_{re.sub(r'[^A-Za-z0-9_-]', '_', session_id)}_rapport.md",
        mime="text/markdown",
    )
    st.download_button(
        "⬇ Télécharger le JSON de cette session",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"BMS_{re.sub(r'[^A-Za-z0-9_-]', '_', session_id)}_diagnosis.json",
        mime="application/json",
    )

st.markdown(
    '<div class="footer">ECU-BMS-Project · Module M3 · Sources : données M1 et sorties de M2. '
    'Les scores ML indiquent des hypothèses, non des causes physiques confirmées. '
    'L’évaluation fonctionnelle EOL reste hors périmètre des données actuelles.</div>',
    unsafe_allow_html=True,
)
