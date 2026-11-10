# Intelligent ECU Diagnosis — BMS Simulation and Machine Learning

An academic prototype for analyzing simulated Battery Management System (BMS) measurements, detecting abnormal operating conditions, classifying known fault patterns, and exporting evidence-based diagnostic reports.

The project is organized into two independently maintained components:

- **Member 1 — BMS simulation:** produces reproducible battery measurements and injects known fault scenarios.
- **Member 2 — Diagnostics and machine learning:** extracts numerical features, evaluates measurement limits, trains anomaly-detection and fault-classification models, and produces structured JSON reports.

This repository covers these two components only. The diagnostic JSON is designed for integration with a separate reporting or AI-agent component.

> **Project status:** Proof of concept using **synthetic BMS data**. The current implementation identifies measurement-limit violations and suspected fault patterns; it does not yet determine functional ECU test PASS/FAIL from actuator responses or timing requirements.

## How it works

```text
BMS simulation (Member 1)
    |
    +--> train.csv / test.csv
                |
                v
       Data validation + feature extraction
                |
                +--> Random Forest: classify known fault types
                +--> Isolation Forest: flag unusual time windows
                +--> Rule engine: check configured measurement limits
                                |
                                v
                       Diagnostic JSON + evidence
                                |
                                v
                     Downstream reporting / AI agent
```

The machine-learning models and the rule engine play different roles. A rule violation indicates that a measurement crossed a configured limit; an anomaly score indicates unusual behavior relative to a learned baseline; and a class prediction identifies a fault pattern seen during training. **None of these alone proves an ECU functional failure or a physical root cause.**

## Repository structure

```text
ECU_BMS_Team_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── member1_simulator/
│   ├── README.md
│   ├── python/generate_dataset.py
│   ├── config/battery_config.json
│   └── datasets/
│       ├── all_scenarios.csv
│       ├── train.csv
│       └── test.csv
├── member2_ml/
│   ├── README.md
│   ├── src/
│   │   ├── bms_pipeline.py
│   │   └── normalize_input.py
│   ├── config/
│   │   ├── rules.json
│   │   └── input_mapping.example.json
│   ├── models/
│   ├── results/
│   └── tests/
└── docs/
    └── data_contract.md
```

See the component READMEs for implementation details: [BMS simulator](member1_simulator/README.md) and [ML diagnostics](member2_ml/README.md). The expected exchange format is documented in [the data contract](docs/data_contract.md).

## Getting started

**Prerequisite:** Python 3.10 or newer is recommended. Run the following commands from the **repository root**.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Generate BMS data (optional)

```bash
python member1_simulator/python/generate_dataset.py
```

The repository already includes the generated dataset, so the remaining steps can be run without regenerating it.

### 3. Inspect session-level features

```bash
python member2_ml/src/bms_pipeline.py features \
  --input member1_simulator/datasets/train.csv \
  --out member2_ml/results/train_session_features.csv
```

> In Windows PowerShell, enter multiline examples on one line or use PowerShell backticks instead of the Bash `\` continuations shown here.

### 4. Train the models

```bash
python member2_ml/src/bms_pipeline.py train
```

This fits a Random Forest classifier and an Isolation Forest anomaly detector, and writes the trained artifacts and training metadata to `member2_ml/models/`.

### 5. Evaluate on held-out simulations

```bash
python member2_ml/src/bms_pipeline.py evaluate
```

The evaluation generates `heldout_metrics.json`, `heldout_predictions.csv`, and `heldout_diagnoses.json` under `member2_ml/results/`.

### 6. Generate diagnostic JSON

For all held-out sessions:

```bash
python member2_ml/src/bms_pipeline.py predict \
  --input member1_simulator/datasets/test.csv \
  --out member2_ml/results/diagnoses.json
```

For one session:

```bash
python member2_ml/src/bms_pipeline.py predict \
  --input member1_simulator/datasets/test.csv \
  --session-id F02_R002 \
  --out member2_ml/results/example_undervoltage_diagnosis.json
```

### 7. Run the automated tests

```bash
python -m pytest -q member2_ml/tests
```

## Data and fault classes

The current dataset contains **160 independent simulated sessions**: 128 for training and 32 for held-out evaluation. Each session records 121 time-stamped samples of battery-pack voltage, battery-pack current, temperature, and state of charge (SOC).

The eight generated conditions are:

| Class ID | Condition |
|---|---|
| `F00` | Normal operation |
| `F01` | Overvoltage |
| `F02` | Undervoltage |
| `F03` | Overcurrent |
| `F04` | Overtemperature |
| `F05` | Voltage-sensor fault |
| `F06` | Current-sensor fault |
| `F07` | Temperature-sensor fault |

Fault labels originate from the simulator and are used for training and evaluation; they are excluded from the predictive feature set.

## Prototype evaluation

On the **32 held-out sessions from the same synthetic generator**:

| Model | Metric | Observed result |
|---|---|---|
| Random Forest | Fault classification accuracy | 32/32 (100%) |
| Isolation Forest | Fault detection recall | 22/28 (78.6%) |
| Isolation Forest | Fault detection precision | 22/22 (100%) |

These results demonstrate that the implementation works on the provided simulation, **not** that it is accurate on production ECU recordings. The held-out set contains only four sessions per class, and its faults are generated by the same simplified model as the training data.

## Diagnostic output

The `predict` command writes a JSON report containing:

- The analyzed session ID and model/ruleset versions.
- The status of configured measurement limits and timestamped supporting evidence.
- The Random Forest's suspected fault class and uncalibrated class score.
- Isolation Forest anomaly windows.
- Missing information and recommended follow-up checks.

The functional `verdict` is deliberately **`NOT_EVALUATED`**: no ECU protective-command outputs, actuator feedback, or approved response deadlines are included in this dataset. See [Member 2's README](member2_ml/README.md#diagnostic-json-interface) for the interface details.

## Integration and future work

The existing pipeline accepts **CSV measurements** with the schema specified in `docs/data_contract.md`. For differently named CSV columns, `member2_ml/src/normalize_input.py` provides an explicit name-mapping step; physical meaning and units still need verification.

In future iterations, the system can be extended to:

1. Decode `.asc`/`.blf` logs using matching DBC files and convert them into synchronized measurements.
2. Analyze functional ECU responses against scenario-specific test plans and response deadlines.
3. Train and evaluate on more varied, independent simulated or authorized real-world executions.
4. Integrate diagnostic JSON with an AI-agent workflow and an engineering dashboard.

A different ECU or substantially different signal distribution requires an adapted feature configuration and an appropriately retrained or independently validated model.

## Reproducibility and security

Generated `.joblib` model binaries are included in the distributable project archive but ignored by Git; they can be recreated with the `train` command. Serialized models should only be loaded from trusted sources. Real automotive recordings may contain confidential information and should not be committed without authorization.

No physical root-cause confirmation or industrial safety validation is claimed by this prototype.
