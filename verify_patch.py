import json

with open(r"notebooks\01_Clustering_Analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "read_csv" in src or "cfg" in src or "yaml" in src:
            print(f"Cell {i}:")
            print(src[:400])
            print()
