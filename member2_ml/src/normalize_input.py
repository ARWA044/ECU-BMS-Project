"""Optional generic CSV column-name adapter for a future Member 1 data source.

Does NOT decode ASC. Requires matching units/physical meanings verified first.
Example: python src/normalize_input.py --input other.csv --mapping config/input_mapping.example.json --out datasets/normalized.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd

CORE = ["scenario_id", "timestamp_s", "pack_voltage_v", "pack_current_a", "temperature_c", "soc_pct"]


def normalize(src, mapping_file, dst):
    mapping = json.loads(Path(mapping_file).read_text(encoding="utf-8"))["columns"]
    df = pd.read_csv(src)
    missing_sources = [name for name in mapping if name not in df]
    if missing_sources:
        raise ValueError(f"Missing source columns: {missing_sources}")
    dests = list(mapping.values())
    if len(set(dests)) != len(dests):
        raise ValueError("Ambiguous mapping: two inputs point to one destination")
    # Rename instead of constructing new columns so labels remain available
    # for training-only scripts. Prediction discards extra metadata.
    df = df.rename(columns=mapping)
    missing = [c for c in CORE if c not in df]
    if missing:
        raise ValueError(f"Required standardized columns missing: {missing}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return len(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(f"Normalized {normalize(args.input, args.mapping, args.out)} rows to {args.out}")
