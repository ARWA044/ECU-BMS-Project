# Member 1 — Synthetic BMS simulator

**Owner: Member 1.** Generates the measurements and ground truth used to develop and evaluate Member 2's ML pipeline.

## Included files
- `python/generate_dataset.py` — simulates 8 scenario classes, including normal behavior and injected battery/sensor faults.
- `config/battery_config.json` — synthetic battery parameters, illustrative limits, sampling settings, fault onset, random seed and train/test split.
- `datasets/all_scenarios.csv` — all 160 independent runs.
- `datasets/train.csv` — 128 independent runs, fit ML models on this file only.
- `datasets/test.csv` — 32 different runs, only for evaluation.

## Run from the SHARED repository root

```powershell
python member1_simulator/python/generate_dataset.py
```

This script automatically writes into `member1_simulator/datasets/` regardless of where it is launched. **It overwrites** the existing generated CSVs.

## Important details
- Each `scenario_id` is one simulation session with measurements at 1-second intervals for 120 seconds.
- `scenario_fault_id` is the session's injected F00–F07 class; `fault_start_time_s`, `fault_detected`, `root_cause` and `fault_type` are simulator-only labels/metadata. **Do not feed them into the model as predictors.**
- The faulted sessions initially look normal and the injected fault starts at 40 seconds.
- Synthetic battery pack voltage is **not** the ECU's low-voltage power supply. All physics and limits here are illustrative, not production specifications.
- The currently generated CSV does **not** include ECU response commands or test-plan timing requirements, so Member 2 cannot report functional PASS/FAIL yet.

See `../docs/data_contract.md` for future simulation output requirements.
