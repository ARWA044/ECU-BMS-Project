import sys
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT.parent / "member1_simulator" / "datasets"
sys.path.insert(0, str(ROOT / "src"))
import bms_pipeline as bms


def test_session_split_disjoint():
    train = pd.read_csv(DATASETS / "train.csv", usecols=["scenario_id"])
    test = pd.read_csv(DATASETS / "test.csv", usecols=["scenario_id"])
    assert not set(train.scenario_id) & set(test.scenario_id)
    assert train.scenario_id.nunique() == 128
    assert test.scenario_id.nunique() == 32


def test_no_training_metadata_in_features():
    data = pd.read_csv(DATASETS / "train.csv")
    normal = bms.read_measurements(DATASETS / "train.csv")
    assert set(normal.columns) == set(bms.REQUIRED)
    features = bms.session_features(normal)
    for key in ["fault_detected", "fault_type", "fault_type_id", "scenario_fault_id", "root_cause", "fault_start_time_s"]:
        assert key not in features.columns
    assert features.shape[0] == 128


def test_limits_on_known_under_voltage():
    data = bms.read_measurements(DATASETS / "test.csv")
    group = data[data.scenario_id == "F02_R002"]
    rules = json.loads((ROOT / "config/rules.json").read_text())
    hit = bms.evaluate_limits(group, rules)
    assert any(e["rule_id"] == "PACK_UNDERVOLTAGE" for e in hit)
    assert all(e["timestamp_s"] in set(group.timestamp_s) for e in hit)


def test_frozen_predictions_are_available():
    data = bms.read_measurements(DATASETS / "test.csv")
    group = data[data.scenario_id == "F02_R002"]
    report = bms.make_reports(group, ROOT / "models", ROOT / "config/rules.json")[0]
    assert report["verdict"] == "NOT_EVALUATED"
    assert report["diagnostic_status"] in {"suspected", "uncertain", "no_anomaly_detected"}
    assert report["data_source"] == "synthetic_bms"
    assert report["recommended_test_done"] is False
    assert report["measurement_limits_status"] == "LIMIT_VIOLATION"


def test_predictions_do_not_use_hidden_labels(tmp_path):
    data = pd.read_csv(DATASETS / "test.csv")
    sid = "F02_R002"
    one = data[data.scenario_id == sid].copy()
    # Replace the hidden ground truth with the deliberately wrong class;
    # this must not affect the predicted label or evidence.
    altered = one.copy()
    for key in ["scenario_fault_id", "fault_type", "fault_type_id", "fault_detected", "root_cause", "fault_start_time_s"]:
        altered[key] = "FAKE"
    path = tmp_path / "changed_labels.csv"
    altered.to_csv(path, index=False)
    a = bms.make_reports(bms.read_measurements(path), ROOT / "models", ROOT / "config/rules.json")[0]
    original = bms.make_reports(bms.read_measurements(DATASETS / "test.csv").query("scenario_id == @sid"), ROOT / "models", ROOT / "config/rules.json")[0]
    assert a == original


def test_missing_required_signal_fails_safely(tmp_path):
    data = pd.read_csv(DATASETS / "test.csv")
    data = data.drop(columns=["pack_voltage_v"])
    p = tmp_path / "broken.csv"
    data.to_csv(p, index=False)
    try:
        bms.read_measurements(p)
    except ValueError as exc:
        assert "pack_voltage_v" in str(exc)
    else:
        raise AssertionError("Missing signal was not rejected")
