"""
generate_sample_data.py
-----------------------
CLI script to generate and save the synthetic demonstration dataset.

Usage
-----
    python scripts/generate_sample_data.py [--rows 500] [--out data/sample_data.csv]

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_sample_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic product-listing compliance dataset."
    )
    parser.add_argument(
        "--rows", type=int, default=500,
        help="Number of rows to generate (default: 500)."
    )
    parser.add_argument(
        "--out", type=str, default="data/sample_data.csv",
        help="Output CSV file path (default: data/sample_data.csv)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} synthetic listing rows (seed={args.seed})...")
    df = generate_sample_data(n=args.rows, random_seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print("\nColumn list:")
    for col in df.columns:
        print(f"  {col}")
    print(
        "\nNOTE: This is SYNTHETIC DEMONSTRATION data only. "
        "It does not represent verified regulatory evidence."
    )


if __name__ == "__main__":
    main()
