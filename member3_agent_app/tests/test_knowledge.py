import json
from member3_agent_app.knowledge import load_documents, retrieve_references, rules_crosscheck


def test_local_rag_cites_only_present_files(tmp_path):
    (tmp_path / "member2_ml/config").mkdir(parents=True)
    (tmp_path / "member1_simulator/config").mkdir(parents=True)
    (tmp_path / "member2_ml/README.md").write_text(
        "The UNDERVOLTAGE fault is identified when pack voltage falls below a threshold. "
        "It is a simulated diagnostic condition and does not prove ECU functional failure.\n",
        encoding="utf-8",
    )
    (tmp_path / "member2_ml/config/rules.json").write_text(json.dumps({
        "ruleset_version": "bms-synthetic-v1", "rules": [
            {"id": "PACK_UNDERVOLTAGE", "signal": "pack_voltage_v", "op": "<", "limit": 300.0, "unit": "V"}
        ]
    }), encoding="utf-8")
    report = {"detected_fault": "UNDERVOLTAGE", "evidence": [{"signal": "pack_voltage_v", "timestamp_s": 62.0,
                                                               "value": 289.57, "rule_id": "PACK_UNDERVOLTAGE"}],
              "recommended_test": "Verify pack voltage"}
    sources = retrieve_references(report, project_root=tmp_path)
    assert sources and all((tmp_path / s["source"]).is_file() for s in sources)
    assert all(s["start_line"] >= 1 and s["end_line"] >= s["start_line"] for s in sources)
    checks = rules_crosscheck(report, project_root=tmp_path)
    assert len(checks) == 1 and checks[0]["rule_matches_evidence"] is True


def test_no_docs_yields_empty_sources(tmp_path):
    assert load_documents(tmp_path) == []
    assert retrieve_references({"detected_fault": "UNDERVOLTAGE"}, project_root=tmp_path) == []
