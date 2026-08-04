"""Pre-generate all graphs and save them to api/graphs.json.

Run this locally before deploying to Vercel:
    python generate_graphs_static.py

This avoids bundling matplotlib/seaborn/pandas into the serverless function.
"""

import json
from pathlib import Path

# graphs.py lives next to this script
from graphs import generate_graphs

OUTPUT = Path(__file__).resolve().parent / "api" / "graphs.json"


def main():
    print("Generating graphs…")
    charts = generate_graphs()
    OUTPUT.write_text(json.dumps(charts), encoding="utf-8")
    total_kb = OUTPUT.stat().st_size // 1024
    print(f"✓  Saved {len(charts)} charts → {OUTPUT}  ({total_kb} KB)")


if __name__ == "__main__":
    main()
