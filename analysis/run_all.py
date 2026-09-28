from __future__ import annotations

import os
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "00_direction_share.py",
    "01_resolved_sets.py",
    "02_matched_static_quality.py",
    "03_matched_dynamic_performance.py",
    "04_matched_issue_patterns.py",
    "05_selection_bias.py",
    "06_patch_size_analysis.py",
]


def main() -> None:
    env = os.environ.copy()
    matplotlib_dir = HERE / ".matplotlib"
    cache_dir = HERE / ".cache"
    matplotlib_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True, env=env)
    environment = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "scripts": SCRIPTS,
    }
    (HERE / "results" / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    artifacts = sorted((HERE / "results").glob("*")) + sorted((HERE / "figures").glob("*"))
    artifacts = [path for path in artifacts if path.name != "reproducibility_manifest.csv"]
    with (HERE / "results" / "reproducibility_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in artifacts:
            writer.writerow({
                "path": str(path.relative_to(HERE)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    print("\nAll analyses completed.")


if __name__ == "__main__":
    main()
