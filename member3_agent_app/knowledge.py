"""RAG léger, local et vérifiable pour la documentation BMS (sans base vectorielle distante).

- Sources explicitement autorisées dans le dépôt : M1/M2, rules.json et config batterie.
- Chaque extrait possède sa source et ses numéros de lignes.
- Les données de simulation ne sont pas un plan d'essais ECU homologué.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
# Liste blanche : aucun document privé ou fichier aléatoire n'est indexé automatiquement.
DOCUMENTS = [
    "README.md",
    "member1_simulator/README.md",
    "member2_ml/README.md",
    "member1_simulator/config/battery_config.json",
    "member2_ml/config/rules.json",
]


def _chunks(content: str, path: str, max_chars: int = 1050) -> list[dict[str, Any]]:
    lines = content.splitlines()
    chunks: list[dict[str, Any]] = []
    buff: list[str] = []
    start = 1

    def emit(end: int) -> None:
        nonlocal buff
        text = "\n".join(buff).strip()
        if len(text) >= 35:
            chunks.append({"source": path, "start_line": start, "end_line": end, "text": text})
        buff = []

    for n, line in enumerate(lines, start=1):
        if buff and (sum(len(s) + 1 for s in buff) + len(line) > max_chars):
            emit(n - 1)
            start = n
        if not buff:
            start = n
        buff.append(line)
    if buff:
        emit(len(lines))
    return chunks


def load_documents(project_root: str | Path = ROOT) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    all_chunks = []
    for rel_path in DOCUMENTS:
        source = (root / rel_path).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            continue
        # Do not index arbitrarily large documents.
        if source.stat().st_size > 250_000:
            continue
        raw = source.read_text(encoding="utf-8-sig")
        if source.suffix == ".json":
            # Validate JSON while retaining ORIGINAL line breaks for traceable citations.
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                continue
        all_chunks.extend(_chunks(raw, rel_path))
    return all_chunks


def retrieve_references(diagnosis: dict[str, Any], project_root: str | Path = ROOT, top_k: int = 4) -> list[dict]:
    """Recherche TF-IDF multi-termes; renvoie uniquement des extraits présents sur disque."""
    docs = load_documents(project_root)
    if not docs:
        return []
    signals = [str(ev.get("signal", "")) for ev in (diagnosis.get("evidence") or []) if isinstance(ev, dict)]
    rule_ids = [str(ev.get("rule_id", "")) for ev in (diagnosis.get("evidence") or []) if isinstance(ev, dict)]
    fault = str(diagnosis.get("detected_fault") or "")
    query = " ".join([
        fault, fault.replace("_", " "),
        str(diagnosis.get("measurement_limits_status") or ""),
        " ".join(signals), " ".join(rule_ids),
        str(diagnosis.get("recommended_test") or ""),
        "battery BMS pack measurement limits fault simulation rules recommended test",
    ])
    # Underscores and casing often distinguish signal identifiers: char_wb helps match them.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform([doc["text"] for doc in docs] + [query])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranking = sorted(range(len(docs)), key=lambda i: (-float(similarities[i]), docs[i]["source"], docs[i]["start_line"]))
    results = []
    for i in ranking:
        score = float(similarities[i])
        if score < 0.085:  # reject virtually unrelated excerpts
            continue
        entry = {**docs[i], "relevance": round(score, 3)}
        results.append(entry)
        if len(results) == top_k:
            break
    return results


def rules_crosscheck(diagnosis: dict[str, Any], project_root: str | Path = ROOT) -> list[dict]:
    """Recompute simple configured comparisons for M2's rule-based evidence; not an ECU test verdict."""
    path = Path(project_root) / "member2_ml" / "config" / "rules.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules = {rule["id"]: rule for rule in raw.get("rules", []) if isinstance(rule, dict) and "id" in rule}
    checked = []
    for evidence in (diagnosis.get("evidence") or []):
        if not isinstance(evidence, dict):
            continue
        rule = rules.get(evidence.get("rule_id"))
        if not rule:
            continue
        try:
            val = float(evidence["value"])
            limit = float(rule["limit"])
        except (ValueError, TypeError, KeyError):
            continue
        symbol = rule.get("op")
        computed = (val > limit if symbol == ">" else val < limit if symbol == "<" else None)
        if computed is None:
            continue
        signal_matches = (evidence.get("signal") == rule.get("signal"))
        checked.append({
            "rule_id": rule["id"], "signal": evidence.get("signal"),
            "value": val, "op": symbol, "limit": limit,
            "rule_matches_evidence": bool(computed and signal_matches),
            "source": "member2_ml/config/rules.json",
        })
    return checked
