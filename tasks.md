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

- [/] **1.7 — Verify full notebook re-runs cleanly top-to-bottom**
  - Kernel > Restart and Run All
  - Requires manual run in Jupyter — open `notebooks/01_Clustering_Analysis.ipynb`

- [x] **1.8 — Git commit**
  - Message: `refactor: restructure project layout and fix reproducibility`

---

## Day 2 — Expand the Algorithms (Technical Depth)
**Theme:** IBM tests algorithm selection judgment. K-Means alone is a homework answer. This day adds two algorithms that demonstrate you understand the full landscape.

- [ ] **2.1 — Add DBSCAN to the notebook**
  - Import `DBSCAN` from `sklearn.cluster`
  - Run a parameter search over `eps` values (0.3, 0.5, 0.8) and `min_samples` (5, 10)
  - Report number of clusters found and noise points identified
  - Compute Silhouette Score (excluding noise label -1)

- [ ] **2.2 — Add Agglomerative Hierarchical Clustering**
  - Import `AgglomerativeClustering` from `sklearn.cluster`
  - Import `dendrogram`, `linkage` from `scipy.cluster.hierarchy`
  - Plot the dendrogram (truncated to last 20 merges for readability)
  - Run with `n_clusters=2` and `n_clusters=3`, compute all three metrics

- [ ] **2.3 — Build the Algorithm Comparison Table**
  - Create a single DataFrame comparing K-Means, DBSCAN, and Agglomerative Hierarchical
  - Columns: Algorithm, Best K, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
  - Display as a formatted table in the notebook

- [ ] **2.4 — Write the Algorithm Selection Rationale (markdown cell)**
  - Explain in plain English why K-Means was chosen as the final model
  - Cover: scaled data, interpretable centroids, business need for k-specific pricing tiers
  - Mention when DBSCAN would be the appropriate choice instead (unknown k, non-convex shapes)

- [ ] **2.5 — Add the per-sample Silhouette Plot**
  - Use `sklearn.metrics.silhouette_samples` to get per-point scores
  - Plot horizontal bar chart coloured by cluster
  - Save to `reports/figures/silhouette_plot.png`

- [ ] **2.6 — Git commit**
  - Message: `feat: add DBSCAN and hierarchical clustering with algorithm comparison table`

---

## Day 3 — Business Narrative and Advanced Visuals (The IBM Differentiator)
**Theme:** IBM hires data scientists, not data mechanics. This day proves you can turn cluster labels into a pricing strategy.

- [ ] **3.1 — Profile each cluster (feature means per segment)**
  - `cluster_profiles = df.groupby('cluster').mean()`
  - Display the full profile table

- [ ] **3.2 — Name and define the driver personas**
  - Give each cluster a business label (e.g., "Cautious Commuters", "High-Risk Urban Drivers")
  - Write a 2-3 sentence profile for each segment explaining the driving behaviour pattern

- [ ] **3.3 — Build the Pricing Implications Table**
  - Create a markdown table in the notebook:
    - Columns: Segment, Label, Key Traits, Estimated Risk Level, Recommended Pricing Action
  - This is the deliverable a client-facing IBM team would actually present

- [ ] **3.4 — Build the Cluster Radar / Spider Chart**
  - Use `matplotlib` with polar axes to plot feature values per cluster
  - Normalise feature values (0-1) so all features sit on the same scale
  - Each cluster = one coloured line on the radar
  - Save to `reports/figures/cluster_radar.png`

- [ ] **3.5 — Build the Feature Heatmap by Cluster**
  - Use `seaborn.heatmap` on cluster centroids
  - Rows = clusters, Columns = features
  - Use a diverging colormap (coolwarm)
  - Save to `reports/figures/cluster_heatmap.png`

- [ ] **3.6 — Ensure all figures are saved to `reports/figures/`**
  - Elbow plot: `elbow_method.png`
  - Silhouette plot: `silhouette_plot.png`
  - PCA scatter: `pca_clusters.png`
  - Dendrogram: `dendrogram.png`
  - Radar chart: `cluster_radar.png`
  - Heatmap: `cluster_heatmap.png`

- [ ] **3.7 — Git commit**
  - Message: `feat: add cluster personas, pricing narrative, radar chart, and heatmap`

---

## Day 4 — Responsible AI, MLOps Signal, and README
**Theme:** These details place a candidate in the top 1%. Responsible AI is IBM's brand identity. A strong README is the 6-second first impression.

- [ ] **4.1 — Add the Responsible AI section to the notebook**
  - New markdown section titled: `Fairness and Responsible AI Considerations`
  - Address: potential proxy features (location as a proxy for socioeconomic status)
  - Address: pricing discrimination risk from cluster-based premium setting
  - Recommendation: periodic cluster re-evaluation as driver behaviour changes over time
  - Note: GDPR-style governance considerations for telematics data collection

- [ ] **4.2 — Add model serialisation with `joblib`**
  - Save the final K-Means model: `joblib.dump(kmeans, '../reports/kmeans_k2_final.joblib')`
  - Write a `predict_segment()` function that loads the model and scores new driver data
  - This signals deployment thinking, not just experimentation

- [ ] **4.3 — Rewrite `README.md`**
  - Add tech stack badges: Python, scikit-learn, pandas, Jupyter, Streamlit
  - Write a business context paragraph in plain, non-technical language
  - Add a Results at a Glance table (best model, best k, key metric scores)
  - Add a Cluster Profiles summary section with the driver personas
  - Add a How to Reproduce section with exact commands
  - Embed the radar chart image so it renders on GitHub
  - Add a Responsible AI notice

- [ ] **4.4 — Final quality check of the full notebook**
  - Remove any scratch cells or commented-out experiments
  - Ensure every code cell has at least one comment explaining intent
  - Ensure every major section has a markdown header and an interpretation cell below the output
  - Kernel > Restart and Run All — confirm zero errors

- [ ] **4.5 — Git commit**
  - Message: `feat: add Responsible AI section, model serialisation, and rewrite README`

---

## Day 5 — Streamlit Deployment
**Theme:** Deploy an interactive dashboard so IBM can see the clusters live without opening a notebook. This is the detail that makes your project feel production-grade.

- [ ] **5.1 — Design the Streamlit app structure**
  - Single-page app at `app/streamlit_app.py`
  - Sections: Project Overview, Cluster Explorer, Algorithm Comparison, Driver Persona Profiles, Predict New Driver

- [ ] **5.2 — Build the Project Overview section**
  - Display the business context paragraph
  - Show key metrics (best k, Silhouette Score, number of drivers per cluster)
  - Use `st.metric()` cards for the headline numbers

- [ ] **5.3 — Build the Cluster Explorer section**
  - Interactive PCA scatter plot using `plotly.express.scatter`
  - Colour by cluster assignment
  - Hover tooltip showing all feature values for each data point
  - Sidebar slider to toggle between k=2 and k=3

- [ ] **5.4 — Build the Algorithm Comparison section**
  - Display the comparison table (K-Means vs DBSCAN vs Hierarchical)
  - Use `st.dataframe()` with column highlighting on the best-performing values

- [ ] **5.5 — Build the Driver Persona Profiles section**
  - Display the radar chart for each segment
  - Show the pricing implication table beneath each persona card
  - Use `st.columns()` to lay out personas side by side

- [ ] **5.6 — Build the Predict New Driver section**
  - Sidebar inputs for the 5 driving features (sliders)
  - Load the saved `joblib` model
  - On submit, predict and display which cluster the driver belongs to
  - Show the persona label, risk level, and recommended pricing action

- [ ] **5.7 — Add `app/requirements.txt` or consolidate into root `requirements.txt`**
  - Ensure `streamlit`, `plotly`, `joblib`, `pyyaml` are all included

- [ ] **5.8 — Add `README.md` section: How to Run the Dashboard**
  - ```bash
    streamlit run app/streamlit_app.py
    ```
  - Add a live demo link if deploying to Streamlit Community Cloud

- [ ] **5.9 — Deploy to Streamlit Community Cloud**
  - Connect GitHub repo to share.streamlit.io
  - Set entry point to `app/streamlit_app.py`
  - Confirm the live URL works and add it to the README

- [ ] **5.10 — Git commit**
  - Message: `feat: add Streamlit dashboard with cluster explorer and live prediction`

---

## End State Checklist

Before marking this project complete, verify all of the following:

- [ ] Notebook runs end-to-end with zero errors after a fresh kernel restart
- [ ] No hardcoded absolute paths anywhere in the notebook
- [ ] `requirements.txt` present and correct
- [ ] Three algorithms implemented and compared in a single table
- [ ] Cluster personas with pricing implications written up
- [ ] At least 4 figures saved to `reports/figures/` including the radar chart
- [ ] Responsible AI section present in the notebook
- [ ] `joblib` model saved and a `predict_segment()` function exists
- [ ] README is detailed, visual, and communicates the business story clearly
- [ ] Repository is structured: `data/`, `notebooks/`, `src/`, `reports/`, `app/`
- [ ] Streamlit app runs locally with zero errors
- [ ] Live Streamlit deployment URL is confirmed and added to README

---

*Last updated: Day 1 complete — Day 2 in progress*
