# Analysis script

`run_analysis.py` is the authoritative executable workflow.

From the repository root:

```bash
python scripts/run_analysis.py
```

Optional paths:

```bash
python scripts/run_analysis.py \
  --raw-data data/raw/techport_projects_batch1.csv \
  --output-dir outputs \
  --figure-dir figures
```

The script uses a fixed seed, recreates all generated CSV and figure files, and writes their SHA-256 hashes to `outputs/run_manifest.json`.
