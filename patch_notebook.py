"""
patch_notebook.py
-----------------
Day 1 patch script.
- Fixes hardcoded absolute path -> relative path using config.yaml
- Adds config.yaml loading cell at the top
- Inserts sys.path setup so src/ module imports work
Run from the repo root.
"""

import json
import re

NB_PATH = r"notebooks\01_Clustering_Analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ------------------------------------------------------------------ #
# 1. Fix hardcoded CSV path in any code cell                         #
# ------------------------------------------------------------------ #
HARDCODED = re.compile(
    r"""pd\.read_csv\s*\(\s*r?["'][^"']*driver_behavior\.csv["']\s*\)"""
)
REPLACEMENT = "pd.read_csv(cfg['data']['path'])"

patched = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if HARDCODED.search(src):
        new_src = HARDCODED.sub(REPLACEMENT, src)
        cell["source"] = [new_src]
        patched += 1

print(f"Patched {patched} cell(s) with hardcoded path.")

# ------------------------------------------------------------------ #
# 2. Prepend a setup cell (config + sys.path) right after header     #
# ------------------------------------------------------------------ #
SETUP_CELL_SOURCE = """\
import os
import sys
import yaml

# Make src/ importable from the notebook
sys.path.insert(0, os.path.abspath("../src"))

# Load all project settings from config.yaml
with open("../config.yaml", "r") as _f:
    cfg = yaml.safe_load(_f)

print("Config loaded. Data path:", cfg['data']['path'])
"""

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [SETUP_CELL_SOURCE],
}

# Insert after Cell 0 (the title markdown cell)
cells.insert(1, setup_cell)
print("Setup cell inserted at position 1.")

# ------------------------------------------------------------------ #
# 3. Write patched notebook back                                      #
# ------------------------------------------------------------------ #
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched and saved.")
