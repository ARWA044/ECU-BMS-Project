"""M3 - Premier workflow LangGraph de diagnostic BMS.

Entrée: un dictionnaire d'une session, produit par M2 (schema_version 1.1).
Sortie: un état contenant des preuves structurées et une synthèse traçable.

Cette étape est DÉTERMINISTE: pas de LLM, pas de RAG et pas de nouveau
verdict de test ECU. Le diagnostic original de M2 n'est pas modifié.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END


class DiagnosticState(TypedDict, total=False):
    diagnosis: dict[str, Any]
    session_id: str
    evidence: list[dict[str, Any]]
    probable_causes: list[dict[str, Any]]
    observations: list[str]
    cautions: list[str]
    recommended_test: str
    report_text: str


def load_context(state: DiagnosticState) -> dict:
    """Nœud 1: contrôler le contrat et extraire les faits produits par M2."""
    d = state["diagnosis"]
    if not isinstance(d, dict):
        raise ValueError("Le diagnostic doit être un objet JSON (dictionnaire Python).")
    for field in ("session_id", "verdict", "evidence"):
        if field not in d:
            raise ValueError(f"Champ obligatoire manquant : {field}")
    if not isinstance(d["evidence"], list):
        raise ValueError("Le champ evidence doit être une liste.")
    if d.get("schema_version") != "1.1":
        raise ValueError("Le workflow attend schema_version 1.1.")
    causes = d.get("probable_causes") or []
    if not isinstance(causes, list):
        raise ValueError("Le champ probable_causes doit être une liste.")
    return {
        "session_id": str(d["session_id"]),
        "evidence": list(d["evidence"]),
        "probable_causes": list(causes),
        "recommended_test": str(d.get("recommended_test") or "Aucun test documenté"),
    }


def analyze_evidence(state: DiagnosticState) -> dict:
    """Nœud 2: résumer les sorties M2 sans créer de mesures."""
    d = state["diagnosis"]
    anomaly = d.get("anomaly_detection") or {}
    obs = [
        f"Verdict fonctionnel M2 : {d.get('verdict', 'inconnu')}",
        f"Limites de mesure : {d.get('measurement_limits_status', 'non indiqué')}",
        f"Défaut proposé : {d.get('detected_fault') or 'aucun défaut retenu'}",
        f"Statut du diagnostic : {d.get('diagnostic_status', 'non indiqué')}",
        f"Preuves horodatées : {len(state['evidence'])}",
    ]
    if anomaly:
        obs.append(
            "Isolation Forest : "
            + ("anomalie détectée" if anomaly.get("anomaly_detected") is True else "anomalie non détectée")
            + f" ; {anomaly.get('anomalous_windows', '?')} fenêtre(s) sur {anomaly.get('total_windows', '?')}"
        )
    return {"observations": obs}


def verify_uncertainty(state: DiagnosticState) -> dict:
    """Nœud 3: rendre explicites les limites du diagnostic et les tests non faits."""
    d = state["diagnosis"]
    cautions: list[str] = []
    if d.get("verdict") == "NOT_EVALUATED":
        cautions.append("Le PASS/FAIL fonctionnel ECU n'a pas été évalué.")
    if d.get("detected_fault") and d.get("diagnostic_status") != "confirmed":
        cautions.append("La cause physique n'est pas confirmée par ce résultat.")
    if d.get("recommended_test_done") is not True:
        cautions.append("Le contrôle recommandé n'est pas déclaré effectué.")
    for cause in state.get("probable_causes", []):
        if cause.get("score_interpretation") == "uncalibrated_class_probability":
            cautions.append("Le score Random Forest est non calibré et ne prouve pas la cause racine.")
            break
    for info in d.get("missing_info") or []:
        cautions.append(f"Information manquante (M2) : {info}")
    return {"cautions": cautions}


def build_report(state: DiagnosticState) -> dict:
    """Nœud 4: générer une synthèse factuelle, sans LLM."""
    d = state["diagnosis"]
    lines = [
        f"# Synthèse BMS — {state['session_id']}",
        "",
        "## Observations issues du moteur M2",
        *[f"- {item}" for item in state["observations"]],
        "",
        "## Preuves horodatées",
    ]
    if not state["evidence"]:
        lines.append("- Aucune preuve de mesure enregistrée pour cette session.")
    else:
        for ev in state["evidence"]:
            signal = ev.get("signal", "inconnu")
            t = ev.get("timestamp_s", "?")
            value = ev.get("value", "?")
            unit = ev.get("unit") or ""
            support = ev.get("supports") or "Aucune interprétation associée"
            lines.append(f"- t={t} s : {signal} = {value} {unit} — {support}")
    lines += ["", "## Hypothèses communiquées par M2"]
    if not state["probable_causes"]:
        lines.append("- Aucune cause proposée.")
    else:
        for c in state["probable_causes"]:
            method = c.get("method") or "inconnue"
            score = c.get("score")
            score_txt = "non disponible" if score is None else str(score)
            lines.append(f"- {c.get('code', 'inconnue')} ; méthode : {method} ; score M2 : {score_txt}.")
    lines += [
        "", "## Vérification recommandée (M2)",
        state["recommended_test"],
        "", "## Limites et informations manquantes",
        *[f"- {s}" for s in state["cautions"]],
        "", "*Synthèse structurée sans LLM ni RAG. Mesures et hypothèses provenant uniquement de M2.*",
    ]
    return {"report_text": "\n".join(lines)}


def create_graph():
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("load_context", load_context)
    workflow.add_node("analyze_evidence", analyze_evidence)
    workflow.add_node("verify_uncertainty", verify_uncertainty)
    workflow.add_node("build_report", build_report)
    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "verify_uncertainty")
    workflow.add_edge("verify_uncertainty", "build_report")
    workflow.add_edge("build_report", END)
    return workflow.compile()


GRAPH = create_graph()


def run_diagnostic_workflow(diagnosis: dict[str, Any]) -> DiagnosticState:
    """Point d'entrée réutilisable depuis Streamlit ou des tests Python."""
    return GRAPH.invoke({"diagnosis": diagnosis})


if __name__ == "__main__":
    from member3_agent_app.services.data_loader import load_diagnoses

    diagnoses = load_diagnoses()
    chosen = next((d for d in diagnoses if d.get("session_id") == "F02_R002"), diagnoses[0])
    result = run_diagnostic_workflow(chosen)
    print(result["report_text"])
