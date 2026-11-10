"""
Synthetic BMS dataset generator.
Creates scenario-grouped train/test CSV files for fault detection,
fault classification, and root-cause diagnosis.

Illustrative educational data only; not validated battery or ECU data.
Run from the project root:
    python python/generate_dataset.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "battery_config.json"
OUTPUT_DIR = ROOT / "datasets"

FAULTS = {
    "F00": ("NORMAL", "No injected fault"),
    "F01": ("OVERVOLTAGE", "Excessive charging voltage"),
    "F02": ("UNDERVOLTAGE", "Excessive discharge or insufficient cell voltage"),
    "F03": ("OVERCURRENT", "Electrical load exceeds the illustrative current limit"),
    "F04": ("OVERTEMPERATURE", "Excessive thermal load or inadequate cooling"),
    "F05": ("VOLTAGE_SENSOR_FAULT", "Voltage sensor bias or measurement failure"),
    "F06": ("CURRENT_SENSOR_FAULT", "Current sensor bias or measurement failure"),
    "F07": ("TEMPERATURE_SENSOR_FAULT", "Temperature sensor bias or measurement failure"),
}


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_one_scenario(fault_id, repetition, cfg, rng):
    pack = cfg["pack"]
    sim = cfg["simulation"]
    thresholds = cfg["illustrative_thresholds"]

    dt = float(sim["sample_time_s"])
    duration = int(sim["duration_s"])
    fault_start = int(sim["fault_start_s"])
    time = np.arange(0, duration + dt, dt)

    # Vary the normal operating profile slightly between scenarios.
    initial_soc = np.clip(
        pack["initial_soc_pct"] + rng.normal(0, 2.0), 65, 95
    )
    base_current = rng.uniform(8, 35)
    current_true = base_current + 4 * np.sin(time / 13) + rng.normal(0, 0.8, len(time))
    current_true = np.maximum(current_true, 0)

    temp_true = 25 + 0.025 * time + rng.normal(0, 0.25, len(time))
    soc = np.empty(len(time), dtype=float)
    soc[0] = initial_soc

    # Simplified pack voltage signal with an illustrative resistive drop.
    voltage_true = (
        pack["nominal_voltage_v"]
        + 0.45 * (initial_soc - 80)
        - current_true * pack["internal_resistance_ohm"]
        + rng.normal(0, 0.25, len(time))
    )

    active = time >= fault_start
    fault_name, root_cause = FAULTS[fault_id]

    # Inject one controlled fault after the normal pre-fault period.
    if fault_id == "F01":
        voltage_true[active] += 65 + rng.normal(0, 1.5, active.sum())
    elif fault_id == "F02":
        voltage_true[active] -= 65 + rng.normal(0, 1.5, active.sum())
    elif fault_id == "F03":
        current_true[active] = rng.uniform(170, 195) + rng.normal(0, 2, active.sum())
        voltage_true[active] = (
            pack["nominal_voltage_v"]
            - current_true[active] * pack["internal_resistance_ohm"]
            + rng.normal(0, 0.4, active.sum())
        )
    elif fault_id == "F04":
        temp_true[active] += 40 + rng.normal(0, 1.0, active.sum())
    elif fault_id == "F05":
        # Measurement bias: physical voltage remains normal, measured voltage shifts.
        pass
    elif fault_id == "F06":
        # Measurement bias: physical current remains normal, measured current shifts.
        pass
    elif fault_id == "F07":
        # Measurement bias: physical temperature remains normal, measured temperature shifts.
        pass

    voltage_measured = voltage_true.copy()
    current_measured = current_true.copy()
    temp_measured = temp_true.copy()

    if fault_id == "F05":
        voltage_measured[active] += rng.choice([-1, 1]) * rng.uniform(25, 45)
    elif fault_id == "F06":
        current_measured[active] += rng.choice([-1, 1]) * rng.uniform(35, 60)
    elif fault_id == "F07":
        temp_measured[active] += rng.choice([-1, 1]) * rng.uniform(15, 25)

    # Coulomb-counting SOC based on measured current. This is deliberately
    # simplified and should not be treated as a production SOC estimator.
    capacity_as = float(pack["capacity_ah"]) * 3600.0
    for i in range(1, len(time)):
        soc[i] = soc[i - 1] - current_measured[i - 1] * dt / capacity_as * 100.0
    soc = np.clip(soc, 0, 100)

    is_fault = (time >= fault_start).astype(int) if fault_id != "F00" else np.zeros(len(time), dtype=int)
    row_fault = np.where(is_fault == 1, fault_id, "F00")
    row_name = np.where(is_fault == 1, fault_name, "NORMAL")
    row_cause = np.where(is_fault == 1, root_cause, "No injected fault")

    scenario_id = f"{fault_id}_R{repetition:03d}"

    return pd.DataFrame({
        "scenario_id": scenario_id,
        "scenario_fault_id": fault_id,
        "timestamp_s": time,
        "pack_voltage_v": voltage_measured,
        "pack_current_a": current_measured,
        "temperature_c": temp_measured,
        "soc_pct": soc,
        "fault_detected": is_fault,
        "fault_type_id": row_fault,
        "fault_type": row_name,
        "root_cause": row_cause,
        "fault_start_time_s": np.where(is_fault == 1, fault_start, -1),
    })


def main():
    cfg = load_config()
    sim = cfg["simulation"]
    rng = np.random.default_rng(int(sim["random_seed"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    repetitions = int(sim["repetitions_per_fault"])

    for fault_id in FAULTS:
        for rep in range(repetitions):
            frames.append(make_one_scenario(fault_id, rep, cfg, rng))

    data = pd.concat(frames, ignore_index=True)

    # Split by whole scenario, never by individual rows, to reduce time-series leakage.
    scenario_table = data[["scenario_id", "scenario_fault_id"]].drop_duplicates()
    train_ids = []
    test_ids = []
    fraction = float(sim["train_fraction"])

    for fault_id, group in scenario_table.groupby("scenario_fault_id"):
        ids = group["scenario_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_train = max(1, int(round(len(ids) * fraction)))
        train_ids.extend(ids[:n_train])
        test_ids.extend(ids[n_train:])

    train = data[data["scenario_id"].isin(train_ids)].copy()
    test = data[data["scenario_id"].isin(test_ids)].copy()

    data.to_csv(OUTPUT_DIR / "all_scenarios.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Dataset generation complete.")
    print(f"Rows total: {len(data):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Scenarios: {data['scenario_id'].nunique()}")
    print("\nFault samples in all_scenarios.csv:")
    print(data.groupby(["scenario_fault_id"])["scenario_id"].nunique())
    print(f"\nCSV files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
