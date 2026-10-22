# Analysis script

`reproduce_all.py` is the complete v3 executable workflow.

From the repository root:

```bash
python scripts/reproduce_all.py
```

The descriptive component alone still supports optional paths:

```bash
python scripts/run_analysis.py \
  --raw-data data/raw/techport_projects_batch1.csv \
  --output-dir outputs \
  --figure-dir figures
```

The script uses a fixed seed, recreates all generated CSV and figure files, and writes their SHA-256 hashes to `outputs/run_manifest.json`.

`run_validation_extension.py` adds independent nested shortlists, structural
prediction baselines and 100 label-randomisation controls per outcome. Its
compact and detailed results are in `outputs/validation_extension/`.
`reproduce_all.py` also executes `tests/` and creates the complete revision manifest.
