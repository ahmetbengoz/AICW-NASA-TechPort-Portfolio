#!/usr/bin/env python3
"""Reproduce every v3 analytical result and verify image and output integrity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]

def finalize():
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    from run_analysis import PUBLICATION_FIGURE_STEMS
    stems=PUBLICATION_FIGURE_STEMS+["Figure7_nested_shortlist"]
    for p in [ROOT/"figures"/f"{stem}.png" for stem in stems]:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            if im.mode == "RGBA":
                assert im.getchannel("A").getextrema() == (255, 255), p
            assert im.width > 3000 and im.height > 1000, p
    files = [*sorted((ROOT/"outputs").rglob("*.csv")),
             *[ROOT/"figures"/f"{stem}.{ext}" for stem in stems for ext in ["svg","png"]],
             ROOT/"data/raw/techport_projects_batch1.csv"]
    manifest = {"version": "3.0.0", "date": "2026-10-22", "files": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/"outputs/revision_manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(ROOT/"outputs/complete_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/"outputs").rglob("*.csv")):
            info=zipfile.ZipInfo(str(p.relative_to(ROOT/"outputs")), (2026,10,22,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    print(json.dumps({"status": "verified", "version": "3.0.0", "files": len(files)}))

if __name__ == "__main__":
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    for script in ["run_analysis.py", "run_validation_extension.py"]:
        subprocess.run([sys.executable, str(ROOT/"scripts"/script)], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=ROOT, env=env, check=True)
    finalize()
