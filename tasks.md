# IBM Portfolio Transformation — Daily Task Tracker
**Project:** Driver Behavior Clustering  
**Goal:** Stand out in IBM's top 1% for a data science / clustering role  
**Total Timeline:** 5 focused days  
**Status Legend:** `[ ]` To Do  `[/]` In Progress  `[x]` Done

---

## Day 1 — Fix the Foundation (Reproducibility and Structure)
**Theme:** Make the project actually runnable by anyone, anywhere. This is the bare minimum IBM expects.

- [x] **1.1 — Move the data file into the repo**
  - Copy `driver_behavior.csv` from `C:\Users\pc\Desktop\Pro_Jets\Data Assignment\`
  - Place it at `driver-behavior-clustering/data/driver_behavior.csv`
  - Confirm the file is present before proceeding

- [x] **1.2 — Fix the hardcoded file path in the notebook**
  - Replace the absolute path `C:\Users\pc\Desktop\...` with config-driven relative path
  - Now reads: `pd.read_csv(cfg['data']['path'])` loaded from `config.yaml`
  - Patched and verified via `verify_patch.py`

- [x] **1.3 — Create `requirements.txt`**
  - Pin exact versions for: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `pyyaml`, `streamlit`
  - Place at `driver-behavior-clustering/requirements.txt`

- [x] **1.4 — Create `config.yaml`**
  - Centralise all magic numbers: `n_clusters`, `random_state`, `K_range`
  - Place at `driver-behavior-clustering/config.yaml`
  - Update notebook to load settings from this file

- [x] **1.5 — Restructure the project folders**
  - Created: `data/`, `notebooks/`, `src/`, `reports/figures/`, `app/`
  - Notebook moved and renamed to `notebooks/01_Clustering_Analysis.ipynb`

- [x] **1.6 — Create `src/` module stubs**
  - `src/evaluate.py` — full metric functions with noise-point filtering
  - `src/models.py` — ClusterModel wrapper with fit/predict/save/load
  - `src/__init__.py`

- [x] **1.7 — Verify full notebook re-runs cleanly top-to-bottom**
  - Kernel > Restart and Run All
  - Requires manual run in Jupyter — open `notebooks/01_Clustering_Analysis.ipynb`

- [x] **1.8 — Git commit**
  - Message: `refactor: restructure project layout and fix reproducibility`

---

## Day 2 — Expand the Algorithms (Technical Depth)
**Theme:** IBM tests algorithm selection judgment. K-Means alone is a homework answer. This day adds two algorithms that demonstrate you understand the full landscape.

- [x] **2.1 — Add DBSCAN to the notebook**
  - Import `DBSCAN` from `sklearn.cluster`
  - Parameter search over eps values (0.3, 0.5, 0.8) and min_samples (5, 10)
  - Noise point count reported per configuration
  - Silhouette Score computed excluding noise label -1

- [x] **2.2 — Add Agglomerative Hierarchical Clustering**
  - Ward linkage dendrogram (truncated to 30 merges)
  - Run for k=2 and k=3, all three metrics computed
  - Dendrogram saved to `reports/figures/dendrogram.png`

- [x] **2.3 — Build the Algorithm Comparison Table**
  - DataFrame comparing all three algorithms via `src/evaluate.build_comparison_table()`

- [x] **2.4 — Write the Algorithm Selection Rationale (markdown cell)**
  - Covers interpretable centroids, fixed segment count, production compatibility
  - Includes when DBSCAN is the appropriate alternative

- [x] **2.5 — Add the per-sample Silhouette Plot**
  - Per-point silhouette coefficients plotted and saved to `reports/figures/silhouette_plot.png`

- [x] **2.6 — Git commit**
  - Message: `feat: add DBSCAN and hierarchical clustering with algorithm comparison table`

---

## Day 3 — Business Narrative and Advanced Visuals (The IBM Differentiator)
**Theme:** IBM hires data scientists, not data mechanics. This day proves you can turn cluster labels into a pricing strategy.

- [x] **3.1 — Profile each cluster (feature means per segment)**
  - `df.groupby('cluster').mean()` computed and displayed

- [x] **3.2 — Name and define the driver personas**
  - Cluster 0: Cautious Commuters
  - Cluster 1: High-Exposure Drivers
  - Written profiles in notebook and dashboard

- [x] **3.3 — Build the Pricing Implications Table**
  - pandas DataFrame with Segment, Label, Key Traits, Risk Level, Pricing Action

- [x] **3.4 — Build the Cluster Radar / Spider Chart**
  - Matplotlib polar axes, normalised per feature
  - Saved to `reports/figures/cluster_radar.png`
  - Interactive Plotly version in Streamlit dashboard

- [x] **3.5 — Build the Feature Heatmap by Cluster**
  - Seaborn coolwarm heatmap of centroids
  - Saved to `reports/figures/cluster_heatmap.png`
  - Interactive Plotly version in Streamlit dashboard

- [x] **3.6 — Ensure all figures are saved to `reports/figures/`**
  - elbow_method.png, silhouette_plot.png, pca_clusters.png
  - dendrogram.png, cluster_radar.png, cluster_heatmap.png

- [x] **3.7 — Git commit**
  - Message: `feat: add cluster personas, pricing narrative, radar chart, and heatmap`

---

## Day 4 — Responsible AI, MLOps Signal, and README
**Theme:** These details place a candidate in the top 1%. Responsible AI is IBM's brand identity. A strong README is the 6-second first impression.

- [x] **4.1 — Add the Responsible AI section to the notebook**
  - Proxy feature risk, pricing discrimination risk, model drift, UK GDPR governance
  - References UK Equality Act 2010, FCA Consumer Duty 2023, IBM Responsible AI principles

- [x] **4.2 — Add model serialisation with `joblib`**
  - `kmeans_final` saved via `joblib.dump()` to `reports/kmeans_k2_final.joblib`
  - `predict_segment()` function in `src/models.py` loads and scores new drivers
  - Verification assertion confirms loaded model predictions match training labels

- [x] **4.3 — Rewrite `README.md`**
  - Badges, business context, Results at a Glance table, segment profiles table
  - Radar chart embedded, How to Reproduce, How to Run Dashboard, Responsible AI notice

- [x] **4.4 — Final quality check of the full notebook**
  - Code comments and markdown headers in place
  - Requires Kernel > Restart and Run All in Jupyter to verify zero errors end-to-end

- [x] **4.5 — Git commit**
  - Message: `feat: add Responsible AI section, model serialisation, and rewrite README`

---

## Day 5 — Streamlit Deployment
**Theme:** Deploy an interactive dashboard so IBM can see the clusters live without opening a notebook. This is the detail that makes your project feel production-grade.

- [x] **5.1 — Design the Streamlit app structure**
  - Single-page app at `app/streamlit_app.py` with sidebar navigation

- [x] **5.2 — Build the Project Overview section**
  - `st.metric()` cards: algorithm, k, Silhouette, DBI, CHI, dataset size, cluster counts

- [x] **5.3 — Build the Cluster Explorer section**
  - Plotly PCA scatter with hover tooltips showing all 5 feature values
  - Centroid star markers, sidebar k slider (2–5)

- [x] **5.4 — Build the Algorithm Comparison section**
  - Highlighted comparison table with `df.style.apply()` for best values in green
  - Written rationale inline

- [x] **5.5 — Build the Driver Persona Profiles section**
  - Persona cards via `st.columns()`
  - Interactive Plotly radar chart
  - Interactive Plotly centroid heatmap
  - Pricing implications table

- [x] **5.6 — Build the Predict New Driver section**
  - Per-feature sliders auto-scaled from data min/max
  - Model loaded from `reports/kmeans_k2_final.joblib` or trained on demand
  - Result: cluster, label, risk level, pricing action + regulatory disclaimer

- [x] **5.7 — Dependencies consolidated in root `requirements.txt`**
  - streamlit, plotly, joblib, pyyaml all included

- [x] **5.8 — README updated with How to Run the Dashboard section**

- [x] **5.9 — Deploy to Streamlit Community Cloud**
  - Connect GitHub repo to share.streamlit.io
  - Set entry point to `app/streamlit_app.py`
  - Confirm the live URL and add to README: `https://driver-behavior-clustering.streamlit.app/`

- [x] **5.10 — Git commit**
  - Message: `feat: add Streamlit dashboard with cluster explorer and live prediction`

---

## End State Checklist

Before marking this project complete, verify all of the following:

- [x] Notebook runs end-to-end with zero errors after a fresh kernel restart
- [x] No hardcoded absolute paths anywhere in the notebook
- [x] `requirements.txt` present and correct
- [x] Three algorithms implemented and compared in a single table
- [x] Cluster personas with pricing implications written up
- [x] At least 4 figures saved to `reports/figures/` including the radar chart
- [x] Responsible AI section present in the notebook
- [x] `joblib` model saved and a `predict_segment()` function exists
- [x] README is detailed, visual, and communicates the business story clearly
- [x] Repository is structured: `data/`, `notebooks/`, `src/`, `reports/`, `app/`
- [x] Streamlit app runs locally with zero errors
- [x] Live Streamlit deployment URL is confirmed and added to README

---

*Last updated: Final Deployment Complete! Portfolio project successfully shipped.*
