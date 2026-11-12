"""Tests du contrat M2 et de la logique M3. Sur un poste réel, utilise le vrai LangGraph."""
import importlib.util
import sys
import types
from pathlib import Path

# Uniquement pour l'environnement de test hors-ligne si LangGraph n'est pas installé.
# Sur le poste utilisateur, ce bloc n'est pas exécuté.
if importlib.util.find_spec("langgraph") is None:
    pkg = types.ModuleType("langgraph")
    sub = types.ModuleType("langgraph.graph")
    sub.START = "__start__"
    sub.END = "__end__"
    class _FakeStateGraph:
        def __init__(self, schema):
            self.nodes = {}
            self.edges = {}
        def add_node(self, key, fn):
            self.nodes[key] = fn
        def add_edge(self, a, b):
            self.edges[a] = b
        def compile(self):
            nodes, edges = self.nodes, self.edges
            class _Compiled:
                def invoke(self, state):
                    state = dict(state)
                    node = edges[sub.START]
                    while node != sub.END:
                        state.update(nodes[node](state))
                        node = edges[node]
                    return state
            return _Compiled()
    sub.StateGraph = _FakeStateGraph
    pkg.graph = sub
    sys.modules["langgraph"] = pkg
    sys.modules["langgraph.graph"] = sub

import pytest
from member3_agent_app import graph


def example(fault="UNDERVOLTAGE"):
    return {
        "schema_version": "1.1", "data_source": "synthetic_bms", "session_id": "F02_R002",
        "verdict": "NOT_EVALUATED", "measurement_limits_status": "LIMIT_VIOLATION",
        "detected_fault": fault, "diagnostic_status": "suspected",
        "evidence": [{"signal": "pack_voltage_v", "timestamp_s": 62.0,
                      "value": 289.57088, "unit": "V", "rule_id": "PACK_UNDERVOLTAGE",
                      "supports": "pack_voltage_v < 300.0"}],
        "probable_causes": [{"code": "UNDERVOLTAGE", "score": 0.98,
                             "method": "random_forest", "score_interpretation": "uncalibrated_class_probability"}],
        "anomaly_detection": {"anomaly_detected": True, "anomalous_windows": 5, "total_windows": 12},
        "missing_info": ["ECU response signal unavailable"],
        "recommended_test": "Verify pack voltage independently.",
        "recommended_test_done": False,
    }


def test_suspected_not_claimed_confirmed(monkeypatch):
    monkeypatch.setattr(graph, "retrieve_references", lambda d: [{"source": "member2_ml/config/rules.json", "start_line": 1, "end_line": 4,
                                                                     "text": "PACK_UNDERVOLTAGE pack_voltage_v < 300", "relevance": 0.5}])
    monkeypatch.setattr(graph, "rules_crosscheck", lambda d: [{"rule_id": "PACK_UNDERVOLTAGE", "signal": "pack_voltage_v", "value": 289.57088,
                                                              "op": "<", "limit": 300.0, "rule_matches_evidence": True,
                                                              "source": "member2_ml/config/rules.json"}])
    source = example()
    original = str(source)
    r = graph.run_diagnostic_workflow(source)
    assert source.__str__() == original  # M2's data are not altered
    assert r["session_id"] == "F02_R002"
    assert "NOT_EVALUATED" in r["report_text"]
    assert "289.57088" in r["report_text"]
    assert "member2_ml/config/rules.json" in r["report_text"]
    assert any("non calibré" in x for x in r["cautions"])
    assert not any("PASS validé" in x for x in r["observations"])


def test_workflow_stays_local_and_uses_no_external_narrative(monkeypatch):
    monkeypatch.setattr(graph, "retrieve_references", lambda d: [])
    monkeypatch.setattr(graph, "rules_crosscheck", lambda d: [])
    r = graph.run_diagnostic_workflow(example())
    assert r["report_text"].startswith("# Synthèse BMS")
    assert "llm_status" not in r
    assert "llm_narrative" not in r
    assert "use_llm" not in r


def test_invalid_json_rejected():
    d = example()
    d.pop("evidence")
    with pytest.raises(ValueError, match="evidence"):
        graph.run_diagnostic_workflow(d)


def test_normal_scenario_avoids_invented_fault(monkeypatch):
    d = example(fault=None)
    d["session_id"] = "F00_R017"
    d["evidence"] = []
    d["probable_causes"] = []
    d["diagnostic_status"] = "no_anomaly_detected"
    d["measurement_limits_status"] = "WITHIN_CONFIGURED_LIMITS"
    d["anomaly_detection"] = {"anomaly_detected": False, "anomalous_windows": 0, "total_windows": 12}
    monkeypatch.setattr(graph, "retrieve_references", lambda d: [])
    monkeypatch.setattr(graph, "rules_crosscheck", lambda d: [])
    r = graph.run_diagnostic_workflow(d)
    assert "aucun défaut retenu" in r["report_text"]
    assert "Aucune cause proposée" in r["report_text"]
    assert "NOT_EVALUATED" in r["report_text"]
