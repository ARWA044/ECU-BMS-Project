"""BMS fault-detection prototype built exclusively around measured signals.

Workflow:
  train    : train.csv -> feature engineering -> IF + RF -> joblib files
  evaluate : held-out test.csv -> frozen models -> metrics + report JSON
  predict  : new CSV (including one scenario) -> JSON for LangGraph teammate
  features : measurements CSV -> session and temporal-window features

For input CSV, required columns are scenario_id, timestamp_s, pack_voltage_v,
pack_current_a, temperature_c, soc_pct. Extra label/ground-truth columns are
always ignored at inference and never used as features.

No ECU commands or test-response criteria are present in the supplied dataset,
so the JSON 'verdict' is NOT_EVALUATED; measurement limits are evaluated
separately. Sensor causes are hypotheses and cannot be confirmed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED = ["scenario_id", "timestamp_s", "pack_voltage_v", "pack_current_a", "temperature_c", "soc_pct"]
SIGNALS = ["pack_voltage_v", "pack_current_a", "temperature_c", "soc_pct"]
UNITS = {"pack_voltage_v": "V", "pack_current_a": "A", "temperature_c": "C", "soc_pct": "%"}
LABEL_TO_ID = {
    "NORMAL": "F00", "OVERVOLTAGE": "F01", "UNDERVOLTAGE": "F02", "OVERCURRENT": "F03",
    "OVERTEMPERATURE": "F04", "VOLTAGE_SENSOR_FAULT": "F05", "CURRENT_SENSOR_FAULT": "F06",
    "TEMPERATURE_SENSOR_FAULT": "F07"
}
MODEL_VERSION = "bms-prototype-v1"


def read_measurements(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Configure an adapter for other ECU schemas.")
    if data.empty:
        raise ValueError("Input contains no measurements")
    data = data[REQUIRED].copy()  # Explicit guard against training-label leakage
    if data["scenario_id"].isna().any():
        raise ValueError("scenario_id cannot be empty")
    for col in REQUIRED[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data["timestamp_s"].isna().any():
        raise ValueError("Every row needs a numeric timestamp")
    data = data.sort_values(["scenario_id", "timestamp_s"], kind="stable").reset_index(drop=True)
    if data.duplicated(["scenario_id", "timestamp_s"]).any():
        raise ValueError("Duplicate timestamps within a scenario; resolve log timestamps first")
    for scenario, sub in data.groupby("scenario_id"):
        if len(sub) < 5:
            raise ValueError(f"Scenario {scenario} contains fewer than 5 observations")
        if sub[SIGNALS].notna().sum().sum() == 0:
            raise ValueError(f"No valid sensor measurements for scenario {scenario}")
    return data


def _numeric_features(series: pd.Series, prefix: str) -> dict:
    a = series.to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    keys = ["min", "max", "mean", "std", "range", "p10", "p90", "start_end_delta", "late_early_delta", "max_abs_step", "mean_abs_step"]
    if len(a) == 0:
        return {f"{prefix}_{key}": np.nan for key in keys}
    n_quarter = max(1, len(a) // 4)
    steps = np.diff(a)
    result = {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "range": float(np.max(a) - np.min(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "start_end_delta": float(a[-1] - a[0]),
        "late_early_delta": float(np.mean(a[-n_quarter:]) - np.mean(a[:n_quarter])),
        "max_abs_step": float(np.max(np.abs(steps))) if len(steps) else 0.0,
        "mean_abs_step": float(np.mean(np.abs(steps))) if len(steps) else 0.0,
    }
    return {f"{prefix}_{key}": value for key, value in result.items()}


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for scenario, part in df.groupby("scenario_id", sort=True):
        row = {"scenario_id": str(scenario), "sample_count": int(len(part))}
        gaps = np.diff(part["timestamp_s"].to_numpy(dtype=float))
        row["duration_s"] = float(part["timestamp_s"].iloc[-1] - part["timestamp_s"].iloc[0])
        row["max_sample_gap_s"] = float(max(gaps)) if len(gaps) else np.nan
        for s in SIGNALS:
            row.update(_numeric_features(part[s], s))
        output.append(row)
    return pd.DataFrame(output)


def window_features(df: pd.DataFrame, window_s: float = 10.0, min_samples: int = 5) -> pd.DataFrame:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    outputs = []
    for scenario, part in df.groupby("scenario_id", sort=True):
        origin = float(part["timestamp_s"].iloc[0])
        bins = np.floor((part["timestamp_s"].to_numpy() - origin) / window_s).astype(int)
        part = part.assign(_bin=bins)
        for bin_no, group in part.groupby("_bin"):
            if len(group) < min_samples:
                continue
            start = origin + float(bin_no) * window_s
            row = {"scenario_id": str(scenario), "window_start_s": round(start, 6),
                   "window_end_s": round(start + window_s, 6)}
            for s in SIGNALS:
                row.update(_numeric_features(group[s], s))
            outputs.append(row)
    if not outputs:
        raise ValueError("No complete time windows; shorten window_s or provide longer recording")
    return pd.DataFrame(outputs)


def _model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"scenario_id", "window_start_s", "window_end_s"}]


def _labels_from_training_file(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "scenario_fault_id" not in df or "scenario_id" not in df:
        raise ValueError("Training/evaluation file needs scenario_id and scenario_fault_id for labels")
    labels = {}
    for sid, part in df.groupby("scenario_id"):
        types = part["scenario_fault_id"].dropna().unique()
        if len(types) != 1:
            raise ValueError(f"Scenario {sid} has inconsistent ground-truth labels")
        fault = str(types[0])
        if fault not in LABEL_TO_ID.values():
            raise ValueError(f"Unrecognized training label {fault}")
        labels[str(sid)] = next(key for key, value in LABEL_TO_ID.items() if value == fault)
    return labels


def _fit_pipeline(estimator, scale: bool) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=False))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train(train_file: str | Path, model_dir: str | Path, window_s: float = 10.0) -> dict:
    df = read_measurements(train_file)
    labels = _labels_from_training_file(train_file)
    sf = session_features(df)
    wf = window_features(df, window_s)
    ids = sf["scenario_id"].astype(str).tolist()
    if set(ids) != set(labels):
        raise ValueError("Training feature sessions and label sessions do not match")
    y = [labels[id] for id in ids]
    if len(set(y)) < 2 or "NORMAL" not in y:
        raise ValueError("Training needs normal sessions and at least one known fault class")

    # RF classifier: one row per independent scenario, not one row per time sample.
    feature_cols = _model_columns(sf)
    rf = _fit_pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42), scale=False)
    rf.fit(sf[feature_cols], y)

    # IF anomaly detector: only normal independent sessions. Keep a small,
    # separate normal calibration subset for choosing the anomaly threshold.
    normal_ids = sorted([sid for sid in ids if labels[sid] == "NORMAL"])
    if len(normal_ids) < 5:
        raise ValueError("At least 5 independent normal training sessions are needed for calibration")
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(normal_ids)
    n_calibration = max(1, int(np.ceil(len(shuffled) * 0.25)))
    fit_ids = set(shuffled[n_calibration:])
    calibration_ids = set(shuffled[:n_calibration])
    wcols = _model_columns(wf)
    fit_w = wf[wf["scenario_id"].isin(fit_ids)]
    val_w = wf[wf["scenario_id"].isin(calibration_ids)]
    if len(fit_w) < 5 or len(val_w) < 5:
        raise ValueError("Not enough normal time windows to train/calibrate Isolation Forest")
    iso = _fit_pipeline(IsolationForest(n_estimators=250, random_state=42, contamination="auto"), scale=True)
    iso.fit(fit_w[wcols])
    scores = -iso.decision_function(val_w[wcols])
    cutoff = float(np.quantile(scores, 0.95))

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    classifier_info = {"model": rf, "feature_cols": feature_cols, "kind": "session_classifier",
                       "model_version": MODEL_VERSION, "classes": list(rf.classes_),
                       "trained_scenarios": ids, "sklearn_version": sklearn.__version__}
    anomaly_info = {"model": iso, "feature_cols": wcols, "kind": "window_anomaly",
                    "threshold": cutoff, "window_s": window_s, "min_anomalous_windows": 2,
                    "model_version": MODEL_VERSION, "sklearn_version": sklearn.__version__,
                    "fit_normal_sessions": sorted(fit_ids), "calibration_normal_sessions": sorted(calibration_ids)}
    joblib.dump(classifier_info, model_path / "bms_classifier.joblib")
    joblib.dump(anomaly_info, model_path / "bms_isolation.joblib")
    info = {"model_version": MODEL_VERSION, "training_sessions": len(ids),
            "train_classes": {c: y.count(c) for c in sorted(set(y))},
            "normal_train_windows": len(fit_w), "normal_calibration_windows": len(val_w),
            "anomaly_score_threshold": cutoff, "calibration_quantile": 0.95,
            "note": "Training and calibration use independent sessions; models are synthetic BMS only."}
    (model_path / "training_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_models(model_dir: str | Path) -> tuple[dict, dict]:
    path = Path(model_dir)
    # Never load joblib files from untrusted sources: pickle-based deserialization.
    classifier = joblib.load(path / "bms_classifier.joblib")
    anomaly = joblib.load(path / "bms_isolation.joblib")
    for bundle in [classifier, anomaly]:
        if bundle.get("model_version") != MODEL_VERSION:
            raise ValueError("Model version mismatch")
    return classifier, anomaly


def _ensure_columns(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature schema incompatible: {sorted(missing)}")
    if frame[names].isna().all(axis=None):
        raise ValueError("No valid feature values")
    return frame[names]


def infer_ml(df: pd.DataFrame, classifier: dict, anomaly: dict) -> dict[str, dict]:
    sf = session_features(df)
    wf = window_features(df, float(anomaly["window_s"]))
    pred_labels = classifier["model"].predict(_ensure_columns(sf, classifier["feature_cols"]))
    probs = classifier["model"].predict_proba(_ensure_columns(sf, classifier["feature_cols"]))
    model_classes = classifier["model"].classes_
    # Higher 'score' means more anomalous.
    w_scores = -anomaly["model"].decision_function(_ensure_columns(wf, anomaly["feature_cols"]))
    wf = wf[["scenario_id", "window_start_s", "window_end_s"]].copy()
    wf["anomaly_score"] = w_scores
    wf["is_anomalous"] = wf["anomaly_score"] > float(anomaly["threshold"])
    output = {}
    for idx, row in sf.iterrows():
        sid = str(row["scenario_id"])
        windows = wf[wf["scenario_id"] == sid]
        ranked = sorted(zip(model_classes, probs[idx]), key=lambda x: -x[1])
        flagged = windows[windows["is_anomalous"]]
        output[sid] = {
            "predicted_class": str(pred_labels[idx]),
            "class_probability": float(max(probs[idx])),
            "top_classes": [{"code": str(c), "score": round(float(p), 4)} for c, p in ranked[:3]],
            "anomalous_windows": int(len(flagged)),
            "total_windows": int(len(windows)),
            "anomaly_detected": bool(len(flagged) >= anomaly["min_anomalous_windows"]),
            "anomaly_windows": [
                {"start_s": float(w["window_start_s"]), "end_s": float(w["window_end_s"]),
                 "score": round(float(w["anomaly_score"]), 5)}
                for _, w in flagged.iterrows()
            ],
        }
    return output


def evaluate_limits(group: pd.DataFrame, rules: dict) -> list[dict]:
    evidence = []
    for rule in rules["rules"]:
        signal = rule["signal"]
        if signal not in group.columns:
            continue
        vals = group[signal].dropna()
        if vals.empty:
            continue
        op, threshold = rule["op"], float(rule["limit"])
        if op == ">":
            violating = vals[vals > threshold]
            if violating.empty:
                continue
            observed_idx = violating.idxmax()
        elif op == "<":
            violating = vals[vals < threshold]
            if violating.empty:
                continue
            observed_idx = violating.idxmin()
        else:
            raise ValueError(f"Unsupported rule operator {op}")
        record = group.loc[observed_idx]
        evidence.append({"signal": signal, "timestamp_s": float(record["timestamp_s"]),
                         "value": round(float(record[signal]), 5), "unit": rule["unit"],
                         "supports": f"{rule['id']} measurement-limit violation; {signal} {op} {threshold}",
                         "rule_id": rule["id"]})
    return evidence


def abrupt_change_evidence(group: pd.DataFrame, label: str) -> list[dict]:
    affected = {
        "OVERVOLTAGE": "pack_voltage_v", "UNDERVOLTAGE": "pack_voltage_v",
        "VOLTAGE_SENSOR_FAULT": "pack_voltage_v", "OVERCURRENT": "pack_current_a",
        "CURRENT_SENSOR_FAULT": "pack_current_a", "OVERTEMPERATURE": "temperature_c",
        "TEMPERATURE_SENSOR_FAULT": "temperature_c",
    }.get(label)
    if not affected:
        return []
    x = group[affected]
    deltas = x.diff().abs()
    if deltas.notna().sum() == 0:
        return []
    idx = deltas.idxmax()
    if pd.isna(deltas.loc[idx]):
        return []
    record = group.loc[idx]
    return [{"signal": affected, "timestamp_s": float(record["timestamp_s"]),
             "value": round(float(record[affected]), 5), "unit": UNITS[affected],
             "supports": f"Largest observed one-sample change in {affected}: {float(deltas.loc[idx]):.3f} {UNITS[affected]}; not proof of cause"}]


RECOMMEND = {
    "OVERVOLTAGE": "Verify pack-voltage measurement and charging regulation with independent instrumentation.",
    "UNDERVOLTAGE": "Verify pack-voltage measurement and load/voltage behavior with independent instrumentation.",
    "OVERCURRENT": "Check current measurement and load profile against the approved electrical test plan.",
    "OVERTEMPERATURE": "Check thermal loading, temperature sensing and commanded cooling response.",
    "VOLTAGE_SENSOR_FAULT": "Cross-check the voltage sensor against an independent reference measurement.",
    "CURRENT_SENSOR_FAULT": "Cross-check the current sensor against an independent calibrated measurement.",
    "TEMPERATURE_SENSOR_FAULT": "Compare temperature measurement against an independent calibrated temperature reference.",
    "NORMAL": "Continue with the ECU functional test plan; no injected-fault diagnosis is confirmed."
}


def make_reports(df: pd.DataFrame, model_dir: str | Path, rules_file: str | Path,
                 data_source: str = "synthetic_bms") -> list[dict]:
    classifier, anomaly = load_models(model_dir)
    rules = json.loads(Path(rules_file).read_text(encoding="utf-8"))
    learned = infer_ml(df, classifier, anomaly)
    reports = []
    for sid, group in df.groupby("scenario_id", sort=True):
        prediction = learned[str(sid)]
        label = prediction["predicted_class"]
        probability = prediction["class_probability"]
        flagged = bool(prediction["anomaly_detected"])
        violations = evaluate_limits(group, rules)
        if label == "NORMAL" and not flagged and not violations:
            status = "no_anomaly_detected"
        elif (label != "NORMAL" and probability >= 0.65 and (flagged or violations)):
            status = "suspected"
        else:
            status = "uncertain"
        evidence = violations + abrupt_change_evidence(group, label)
        if label == "NORMAL" and not violations:
            # Avoid adding synthetic evidence for an absence-of-failure conclusion.
            evidence = []
        if label != "NORMAL" and (probability < 0.65 or (not flagged and not violations)):
            candidate_code = None
        else:
            candidate_code = label if label != "NORMAL" else None
        reported = {
            "schema_version": "1.1",
            "data_source": data_source,
            "session_id": str(sid),
            "verdict": "NOT_EVALUATED",
            "measurement_limits_status": "LIMIT_VIOLATION" if violations else "WITHIN_CONFIGURED_LIMITS",
            "detected_fault": candidate_code,
            "probable_causes": ([{"code": label, "score": round(probability, 4),
                                  "method": "random_forest", "score_interpretation": "uncalibrated_class_probability"}]
                                if label != "NORMAL" else []),
            "anomaly_detection": {"method": "isolation_forest", "anomaly_detected": flagged,
                                  "anomalous_windows": prediction["anomalous_windows"],
                                  "total_windows": prediction["total_windows"],
                                  "windows": prediction["anomaly_windows"]},
            "evidence": evidence,
            "diagnostic_status": status,
            "missing_info": [
                "ECU response/actuator signals and approved timing requirements for functional PASS/FAIL",
                "Independent physical measurements required for confirmed root-cause diagnosis"
            ],
            "recommended_test": RECOMMEND.get(label, "Perform targeted investigation of the recorded abnormal behavior."),
            "recommended_test_done": False,
            "model_version": MODEL_VERSION,
            "ruleset_version": rules["ruleset_version"]
        }
        reports.append(reported)
    return reports


def calculate_metrics(test_file: str | Path, model_dir: str | Path, rules_file: str | Path) -> tuple[dict, pd.DataFrame, list[dict]]:
    data = read_measurements(test_file)
    y_true = _labels_from_training_file(test_file)
    clf, anomaly = load_models(model_dir)
    train_sids = set(clf.get("trained_scenarios", []))
    if train_sids & set(y_true):
        raise ValueError("Test data overlaps training sessions; hold out independent sessions")
    prediction = infer_ml(data, clf, anomaly)
    sessions = sorted(prediction)
    actual = [y_true[sid] for sid in sessions]
    predicted = [prediction[sid]["predicted_class"] for sid in sessions]
    class_names = list(clf["classes"])
    matrix = confusion_matrix(actual, predicted, labels=class_names)
    evaluations = pd.DataFrame([{"session_id": sid, "true_label": y_true[sid],
                                 "predicted_label": prediction[sid]["predicted_class"],
                                 "probability": round(prediction[sid]["class_probability"], 5),
                                 "anomaly_detected": prediction[sid]["anomaly_detected"]}
                                for sid in sessions])
    expected_anomaly = np.array([y_true[s] != "NORMAL" for s in sessions], dtype=bool)
    found_anomaly = np.array([prediction[s]["anomaly_detected"] for s in sessions], dtype=bool)
    precision, recall, f1, _ = precision_recall_fscore_support(expected_anomaly, found_anomaly,
                                                                average="binary", zero_division=0)
    metrics = {
        "model_version": MODEL_VERSION,
        "heldout_sessions": len(sessions),
        "classification_accuracy": float(accuracy_score(actual, predicted)),
        "classification_report": classification_report(actual, predicted, labels=class_names,
                                                       output_dict=True, zero_division=0),
        "confusion_matrix_labels": class_names,
        "confusion_matrix": matrix.tolist(),
        "anomaly_detection": {"precision": float(precision), "recall": float(recall), "f1": float(f1),
                              "true_faulty_sessions": int(expected_anomaly.sum()),
                              "predicted_anomalous_sessions": int(found_anomaly.sum())},
        "limitations": [
            "Evaluation uses held-out sessions from the same simplified generator; no real ECU validation.",
            "Random Forest probabilities are not calibrated root-cause confidence scores.",
            "Anomalies are not necessarily functional ECU FAILs; test-plan response signals are unavailable."
        ]
    }
    return metrics, evaluations, make_reports(data, model_dir, rules_file)


def write_json(path: str | Path, value: dict | list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="cmd", required=True)
    t = commands.add_parser("train", help="Train models on labeled independent scenarios")
    t.add_argument("--train", default="member1_simulator/datasets/train.csv")
    t.add_argument("--models", default="member2_ml/models")
    t.add_argument("--window-s", type=float, default=10.0)
    ev = commands.add_parser("evaluate", help="Evaluate on separate held-out scenarios")
    ev.add_argument("--test", default="member1_simulator/datasets/test.csv")
    ev.add_argument("--models", default="member2_ml/models")
    ev.add_argument("--rules", default="member2_ml/config/rules.json")
    ev.add_argument("--out-dir", default="member2_ml/results")
    p = commands.add_parser("predict", help="Predict on new measurements (never uses ground-truth columns)")
    p.add_argument("--input", required=True)
    p.add_argument("--models", default="member2_ml/models")
    p.add_argument("--rules", default="member2_ml/config/rules.json")
    p.add_argument("--out", default="member2_ml/results/diagnoses.json")
    p.add_argument("--session-id", help="Optionally report only one scenario")
    p.add_argument("--data-source", default="synthetic_bms")
    fe = commands.add_parser("features", help="Export learned ML features for inspection")
    fe.add_argument("--input", required=True)
    fe.add_argument("--out", default="member2_ml/results/session_features.csv")
    args = ap.parse_args()
    if args.cmd == "train":
        info = train(args.train, args.models, args.window_s)
        print(json.dumps(info, indent=2))
    elif args.cmd == "evaluate":
        metrics, eval_df, reports = calculate_metrics(args.test, args.models, args.rules)
        path = Path(args.out_dir)
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "heldout_metrics.json", metrics)
        eval_df.to_csv(path / "heldout_predictions.csv", index=False)
        write_json(path / "heldout_diagnoses.json", reports)
        print(f"Held-out sessions={metrics['heldout_sessions']}; RF accuracy={metrics['classification_accuracy']:.3f}")
        print(f"IF recall={metrics['anomaly_detection']['recall']:.3f} precision={metrics['anomaly_detection']['precision']:.3f}")
        print(f"Outputs saved to {path}")
    elif args.cmd == "predict":
        data = read_measurements(args.input)
        if args.session_id:
            data = data[data["scenario_id"] == args.session_id]
            if data.empty:
                raise ValueError(f"Session ID {args.session_id} not found")
        result = make_reports(data, args.models, args.rules, args.data_source)
        write_json(args.out, result[0] if args.session_id else result)
        print(f"Wrote {len(result)} diagnostic report(s) to {args.out}")
    elif args.cmd == "features":
        features = session_features(read_measurements(args.input))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.out, index=False)
        print(f"Wrote {len(features)} sessions and {len(features.columns)} columns to {args.out}")


if __name__ == "__main__":
    cli()
