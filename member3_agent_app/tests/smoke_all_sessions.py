"""Smoke-test all current M2 reports against M1 data using the real M3 workflow.

Run from the ECU-BMS-Project repository root:
    python member3_agent_app/tests/smoke_all_sessions.py

Does not train models, modify data, use a LLM, or access the network.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from member3_agent_app.graph import run_diagnostic_workflow
from member3_agent_app.services.data_loader import load_diagnoses, load_measurements


def main() -> int:
    reports = load_diagnoses()
    measurements = load_measurements()
    if not reports:
        print("FAIL: No M2 diagnosis found")
        return 1

    measured_ids = set(measurements["scenario_id"].astype(str))
    failures = []
    seen = set()
    with_sources = 0
    checked_rules = 0
    statuses = Counter()

    for report in reports:
        sid = str(report.get("session_id", ""))
        try:
            if not sid or sid in seen:
                raise ValueError("Missing or duplicated session_id")
            seen.add(sid)
            if sid not in measured_ids:
                raise ValueError("Session does not exist in M1 test.csv")
            source_copy = copy.deepcopy(report)
            result = run_diagnostic_workflow(report)
            if report != source_copy:
                raise ValueError("The workflow modified the input diagnosis")
            if result.get("session_id") != sid:
                raise ValueError("Wrong session_id in output")
            if not result.get("report_text", "").strip():
                raise ValueError("Empty report")
            if report.get("verdict") == "NOT_EVALUATED" and "NOT_EVALUATED" not in result["report_text"]:
                raise ValueError("Unjustified functional verdict: NOT_EVALUATED missing")

            for citation in result.get("rag_sources", []):
                source = (ROOT / citation["source"]).resolve()
                if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
                    raise ValueError("Missing or out-of-repository RAG source")
                number_of_lines = len(source.read_text(encoding="utf-8-sig").splitlines())
                start = citation["start_line"]
                end = citation["end_line"]
                if not (1 <= start <= end <= number_of_lines):
                    raise ValueError("Invalid RAG line reference")

            with_sources += bool(result.get("rag_sources"))
            checked_rules += len(result.get("rule_checks") or [])
            statuses[str(report.get("diagnostic_status", "unknown"))] += 1
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(f"{sid or '<no id>'}: {exc}")

    print(f"M2 diagnoses:          {len(reports)}")
    print(f"M1 measurement rows:   {len(measurements)}")
    print(f"Processed sessions:    {len(reports) - len(failures)}")
    print(f"Sessions with RAG docs:{with_sources:>5}")
    print(f"Rule comparisons:      {checked_rules}")
    print(f"Diagnosis status:      {dict(statuses)}")

    if failures:
        print("\nFAILED SESSIONS:")
        for error in failures:
            print(" -", error)
        return 1
    print("\nPASS: All available sessions passed M3 integration smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
