"""M3 - Workflow LangGraph BMS : diagnostic M2 -> preuves -> RAG local -> vérifications -> rapport.

Fonctionne entièrement en local, sans clé API ni appel à un modèle externe.
La recherche documentaire ne produit pas de verdict fonctionnel ECU PASS/FAIL.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from member3_agent_app.knowledge import retrieve_references, rules_crosscheck


class DiagnosticState(TypedDict, total=False):
    diagnosis: dict[str, Any]
    session_id: str
    evidence: list[dict[str, Any]]
    probable_causes: list[dict[str, Any]]
    observations: list[str]
    cautions: list[str]
    recommended_test: str
    rag_sources: list[dict[str, Any]]
    rule_checks: list[dict[str, Any]]
    rag_status: str
    report_text: str


def load_context(state: DiagnosticState) -> dict:
    d = state.get("diagnosis")
    if not isinstance(d, dict):
        raise ValueError("Diagnostic M2 invalide : objet JSON attendu.")
    for key in ("session_id", "verdict", "evidence"):
        if key not in d:
            raise ValueError(f"Champ M2 obligatoire absent : {key}")
    if d.get("schema_version") != "1.1":
        raise ValueError("Seul le contrat M2 schema_version 1.1 est accepté.")
    if not isinstance(d["evidence"], list) or not all(isinstance(e, dict) for e in d["evidence"]):
        raise ValueError("evidence doit être une liste d'objets.")
    causes = d.get("probable_causes") or []
    if not isinstance(causes, list) or not all(isinstance(c, dict) for c in causes):
        raise ValueError("probable_causes doit être une liste d'objets.")
    return {
        "session_id": str(d["session_id"]),
        "evidence": d["evidence"],
        "probable_causes": causes,
        "recommended_test": str(d.get("recommended_test") or "Aucun test indiqué par M2."),
    }


def analyze_evidence(state: DiagnosticState) -> dict:
    d = state["diagnosis"]
    a = d.get("anomaly_detection") or {}
    lines = [
        f"Verdict fonctionnel M2 : {d.get('verdict')}",
        f"État des seuils : {d.get('measurement_limits_status', 'non indiqué')}",
        f"Défaut proposé : {d.get('detected_fault') or 'aucun défaut retenu'}",
        f"Statut de diagnostic : {d.get('diagnostic_status', 'non indiqué')}",
        f"Preuves horodatées : {len(state['evidence'])}",
    ]
    if a:
        lines.append(
            "Isolation Forest : "
            + ("anomalie détectée" if a.get("anomaly_detected") is True else "aucune anomalie détectée")
            + f" ; {a.get('anomalous_windows', '?')} fenêtre(s) sur {a.get('total_windows', '?')}"
        )
    return {"observations": lines}


def retrieve_docs(state: DiagnosticState) -> dict:
    docs = retrieve_references(state["diagnosis"])
    return {"rag_sources": docs, "rag_status": "available" if docs else "no_documents"}


def verify_uncertainty(state: DiagnosticState) -> dict:
    d = state["diagnosis"]
    warnings: list[str] = []
    if d.get("verdict") == "NOT_EVALUATED":
        warnings.append("Le PASS/FAIL fonctionnel ECU n'a PAS été évalué.")
    if d.get("detected_fault") is not None:
        warnings.append("La prédiction d'une classe de défaut ne confirme pas une cause physique.")
    if d.get("diagnostic_status") != "confirmed" and d.get("detected_fault"):
        warnings.append("Le statut du défaut est suspecté ou incertain, non confirmé.")
    if d.get("recommended_test_done") is not True:
        warnings.append("Le contrôle recommandé n'a pas été déclaré effectué.")
    if any(c.get("score_interpretation") == "uncalibrated_class_probability" for c in state["probable_causes"]):
        warnings.append("Le score Random Forest est non calibré : ce n'est pas une probabilité de cause racine.")
    for info in d.get("missing_info") or []:
        warnings.append(f"Information manquante selon M2 : {info}")

    checks = rules_crosscheck(d)
    if any(not check["rule_matches_evidence"] for check in checks):
        warnings.append("Attention : une preuve M2 ne satisfait pas sa règle de seuil documentée.")
    if not state.get("rag_sources"):
        warnings.append("Aucune référence technique locale pertinente n'a pu être retrouvée.")
    warnings.append("Les seuils BMS sont illustratifs ; aucun plan d'essai OEM n'est homologué ici.")
    return {"cautions": list(dict.fromkeys(warnings)), "rule_checks": checks}


def _time_sort(ev: dict) -> float:
    try:
        return float(ev.get("timestamp_s") or 0)
    except (TypeError, ValueError):
        return float("inf")


def build_report(state: DiagnosticState) -> dict:
    d = state["diagnosis"]
    lines = [
        f"# Synthèse BMS — {state['session_id']}", "",
        "## Observations issues du moteur M2",
        *[f"- {v}" for v in state["observations"]], "",
        "## Preuves horodatées",
    ]
    if not state["evidence"]:
        lines.append("- Aucune preuve de mesure enregistrée pour cette session.")
    else:
        for ev in sorted(state["evidence"], key=lambda e: _time_sort(e)):
            lines.append(
                f"- t={ev.get('timestamp_s', '?')} s : {ev.get('signal', '?')} = "
                f"{ev.get('value', '?')} {ev.get('unit') or ''} — {ev.get('supports') or 'sans justification'}"
            )
    lines.extend(["", "## Hypothèses de M2"])
    if not state["probable_causes"]:
        lines.append("- Aucune cause proposée.")
    for cause in state["probable_causes"]:
        lines.append(
            f"- {cause.get('code', '?')} · {cause.get('method', '?')} · score : "
            f"{cause.get('score') if cause.get('score') is not None else 'non disponible'} "
            f"({cause.get('score_interpretation', 'interprétation non précisée')})."
        )
    lines += ["", "## Documents retrouvés (RAG local)"]
    if state.get("rag_sources"):
        for item in state["rag_sources"]:
            lines.append(f"- `{item['source']}:L{item['start_line']}-L{item['end_line']}` (pertinence lexicale {item['relevance']})")
    else:
        lines.append("- Aucune source documentaire correspondante.")
    lines += ["", "## Contrôles de cohérence des règles"]
    if not state.get("rule_checks"):
        lines.append("- Aucune preuve vérifiable par une règle documentée.")
    for c in state.get("rule_checks") or []:
        label = "conforme au seuil" if c["rule_matches_evidence"] else "non concordant"
        lines.append(f"- {c['rule_id']} : {c['value']} {c['op']} {c['limit']} — {label} (selon {c['source']}).")
    lines += ["", "## Vérification recommandée par M2", state["recommended_test"], "", "## Limites"]
    lines.extend(f"- {w}" for w in state.get("cautions") or [])
    lines += ["", "*Prototype académique sur mesures BMS synthétiques. Ce rapport ne valide pas un ECU réel, ne confirme pas une cause physique et ne modifie pas le verdict de M2.*"]
    return {"report_text": "\n".join(lines)}


def create_graph():
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("load_context", load_context)
    workflow.add_node("analyze_evidence", analyze_evidence)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("verify_uncertainty", verify_uncertainty)
    workflow.add_node("build_report", build_report)
    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "verify_uncertainty")
    workflow.add_edge("verify_uncertainty", "build_report")
    workflow.add_edge("build_report", END)
    return workflow.compile()


GRAPH = create_graph()


def run_diagnostic_workflow(diagnosis: dict[str, Any]) -> DiagnosticState:
    """Interface de compatibilité avec le dashboard existant."""
    return GRAPH.invoke({"diagnosis": diagnosis})


if __name__ == "__main__":
    from member3_agent_app.services.data_loader import load_diagnoses
    diagnoses = load_diagnoses()
    demo = next((d for d in diagnoses if d.get("session_id") == "F02_R002"), diagnoses[0])
    result = run_diagnostic_workflow(demo)
    print(result["report_text"])
