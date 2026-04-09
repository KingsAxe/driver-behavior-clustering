"""
day3_inject.py
--------------
Injects Day 3 content into the notebook:
  - Cluster profiling (feature means per segment)
  - Driver persona labels and written profiles
  - Pricing implications table
  - Radar / spider chart
  - Feature heatmap by cluster
  - All figures saved to reports/figures/
"""

import json

NB_PATH = r"notebooks\01_Clustering_Analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


# Insert Day 3 cells just before Exercise 2 section
insert_before = None
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Exercise 2" in src and cell["cell_type"] == "markdown":
        insert_before = idx
        break

if insert_before is None:
    insert_before = len(nb["cells"])

print(f"Inserting Day 3 cells before cell index {insert_before}")

day3_cells = [

    md_cell(
        "---\n\n"
        "## Business Narrative: Driver Segment Profiles and Pricing Implications\n\n"
        "Clustering is only valuable when its outputs can drive a decision.\n"
        "This section translates the final K-Means cluster assignments into\n"
        "actionable driver personas that an insurance pricing team can act on."
    ),

    # Cluster profiling
    code_cell(
        "import numpy as np\n\n"
        "# Assign final cluster labels to the dataframe\n"
        "df['cluster'] = final_labels\n\n"
        "# Compute mean feature values per cluster\n"
        "cluster_profiles = df.groupby('cluster').mean().round(4)\n"
        "cluster_sizes = df['cluster'].value_counts().sort_index().rename('driver_count')\n\n"
        "print('Cluster sizes:')\n"
        "print(cluster_sizes)\n"
        "print()\n"
        "print('Cluster feature profiles (mean values):')\n"
        "cluster_profiles"
    ),

    # Personas markdown
    md_cell(
        "### Driver Persona Definitions\n\n"
        "Based on the cluster feature profiles above, each segment is assigned a\n"
        "business label that captures the dominant behavioural pattern.\n\n"
        "---\n\n"
        "**Cluster 0 — Cautious Commuters**\n\n"
        "Drivers in this group exhibit stable, low-variance driving behaviour. "
        "They show lower values on hard-braking and acceleration event frequency, "
        "shorter trip distances, and limited night-time driving. "
        "This profile is consistent with routine urban or suburban commuting under predictable conditions.\n\n"
        "---\n\n"
        "**Cluster 1 — High-Exposure Drivers**\n\n"
        "This segment displays elevated scores across higher-risk behavioural features: "
        "more frequent sharp braking events, higher speed variability, and a greater proportion "
        "of trips occurring during night hours or at extended distances. "
        "This profile suggests a broader range of driving contexts — including highway, "
        "night, and higher-speed environments — that collectively increase exposure risk.\n\n"
        "---\n\n"
        "> These labels are informed by the feature distributions and are illustrative.\n"
        "> In a production deployment, the persona definitions would be validated with\n"
        "> the actuarial and pricing teams before any premium adjustments were applied."
    ),

    # Pricing implications table
    code_cell(
        "# Pricing implications summary\n"
        "personas = {\n"
        "    0: {\n"
        "        'label': 'Cautious Commuters',\n"
        "        'risk_level': 'Low',\n"
        "        'key_traits': 'Low braking events, short trips, daytime driving',\n"
        "        'pricing_action': 'Offer 10-20% loyalty discount; upsell telematics programme',\n"
        "    },\n"
        "    1: {\n"
        "        'label': 'High-Exposure Drivers',\n"
        "        'risk_level': 'Elevated',\n"
        "        'key_traits': 'High speed variance, night driving, frequent braking',\n"
        "        'pricing_action': 'Apply risk-adjusted surcharge; offer telematics opt-in for premium reduction',\n"
        "    },\n"
        "}\n\n"
        "pricing_table = pd.DataFrame([\n"
        "    {\n"
        "        'Segment': k,\n"
        "        'Label': v['label'],\n"
        "        'Key Traits': v['key_traits'],\n"
        "        'Risk Level': v['risk_level'],\n"
        "        'Recommended Pricing Action': v['pricing_action'],\n"
        "    }\n"
        "    for k, v in personas.items()\n"
        "])\n\n"
        "pricing_table"
    ),

    # Radar chart
    md_cell(
        "### Cluster Radar Chart\n\n"
        "The radar chart plots each cluster's normalised mean feature values on a\n"
        "common scale (0 to 1), allowing direct visual comparison of behavioural\n"
        "profiles across all five driving dimensions simultaneously."
    ),

    code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "# Feature names from the data (excluding cluster column)\n"
        "feature_cols = [c for c in df.columns if c != 'cluster']\n"
        "N = len(feature_cols)\n\n"
        "# Normalise centroid values to [0, 1] per feature\n"
        "centroids = df.groupby('cluster')[feature_cols].mean()\n"
        "centroid_norm = (centroids - centroids.min()) / (centroids.max() - centroids.min())\n\n"
        "# Angles for radar chart\n"
        "angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()\n"
        "angles += angles[:1]  # close the polygon\n\n"
        "fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))\n\n"
        "cluster_colors = ['#2563EB', '#DC2626']\n"
        "cluster_labels = [personas[c]['label'] for c in centroids.index]\n\n"
        "for cluster_id, (row, color, label) in enumerate(\n"
        "    zip(centroid_norm.values, cluster_colors, cluster_labels)\n"
        "):\n"
        "    values = row.tolist() + row[:1].tolist()\n"
        "    ax_r.plot(angles, values, 'o-', linewidth=2, color=color, label=label)\n"
        "    ax_r.fill(angles, values, alpha=0.12, color=color)\n\n"
        "ax_r.set_thetagrids(np.degrees(angles[:-1]), feature_cols, fontsize=11)\n"
        "ax_r.set_ylim(0, 1)\n"
        "ax_r.set_title('Driver Segment Behavioural Profiles', fontsize=14, fontweight='bold', pad=20)\n"
        "ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)\n"
        "ax_r.grid(True, alpha=0.4)\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../reports/figures/cluster_radar.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Radar chart saved.')"
    ),

    # Heatmap
    md_cell(
        "### Feature Heatmap by Cluster\n\n"
        "The centroid heatmap provides a complementary view to the radar chart:\n"
        "each cell shows the mean value of a given feature for a given cluster,\n"
        "using colour intensity to highlight differences across segments."
    ),

    code_cell(
        "import seaborn as sns\n\n"
        "# Cluster centroid heatmap\n"
        "centroid_means = df.groupby('cluster')[feature_cols].mean()\n\n"
        "fig_h, ax_h = plt.subplots(figsize=(10, 4))\n"
        "sns.heatmap(\n"
        "    centroid_means,\n"
        "    annot=True,\n"
        "    fmt='.3f',\n"
        "    cmap='coolwarm',\n"
        "    linewidths=0.5,\n"
        "    ax=ax_h,\n"
        "    cbar_kws={'label': 'Mean Feature Value'},\n"
        ")\n"
        "ax_h.set_title('Cluster Centroid Feature Heatmap', fontsize=14, fontweight='bold')\n"
        "ax_h.set_xlabel('Driving Feature', fontsize=12)\n"
        "ax_h.set_ylabel('Cluster', fontsize=12)\n"
        "ax_h.set_yticklabels(\n"
        "    [f'Cluster {i} — {personas[i][\"label\"]}' for i in centroid_means.index],\n"
        "    rotation=0,\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.savefig('../reports/figures/cluster_heatmap.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Heatmap saved.')"
    ),

    # Save PCA and Elbow figures too
    code_cell(
        "# Ensure the elbow and PCA plots are also saved to reports/figures/\n"
        "# (Re-generate if needed — these reference variables from earlier cells)\n\n"
        "# Elbow + Silhouette Score diagnostic (re-plot with save)\n"
        "K_range = cfg['clustering']['k_range']\n"
        "inertias_r = []\n"
        "sil_scores_r = []\n\n"
        "for k in K_range:\n"
        "    km_r = KMeans(n_clusters=k, random_state=cfg['clustering']['random_state'], n_init=cfg['clustering']['n_init'])\n"
        "    km_r.fit(X)\n"
        "    inertias_r.append(km_r.inertia_)\n"
        "    sil_scores_r.append(silhouette_score(X, km_r.labels_))\n\n"
        "fig_e, axes_e = plt.subplots(1, 2, figsize=(14, 5))\n"
        "axes_e[0].plot(K_range, inertias_r, marker='o', linewidth=2, color='steelblue')\n"
        "axes_e[0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')\n"
        "axes_e[0].set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')\n"
        "axes_e[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')\n"
        "axes_e[0].grid(True, alpha=0.3)\n\n"
        "axes_e[1].plot(K_range, sil_scores_r, marker='o', linewidth=2, color='darkorange')\n"
        "axes_e[1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')\n"
        "axes_e[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')\n"
        "axes_e[1].set_title('Silhouette Score by k', fontsize=14, fontweight='bold')\n"
        "axes_e[1].grid(True, alpha=0.3)\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../reports/figures/elbow_method.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Elbow + Silhouette diagnostic saved.')\n\n"
        "# PCA scatter (re-plot with save)\n"
        "from sklearn.decomposition import PCA\n"
        "pca_s = PCA(n_components=2)\n"
        "X_pca_s = pca_s.fit_transform(X)\n\n"
        "fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 5))\n"
        "for ax_p, k_p in zip(axes_p, [2, 3]):\n"
        "    km_p = KMeans(n_clusters=k_p, random_state=cfg['clustering']['random_state'])\n"
        "    lbl_p = km_p.fit_predict(X)\n"
        "    centers_p = pca_s.transform(km_p.cluster_centers_)\n"
        "    ax_p.scatter(X_pca_s[:, 0], X_pca_s[:, 1], c=lbl_p, cmap='viridis', s=30, alpha=0.8, edgecolors='k', linewidth=0.3)\n"
        "    ax_p.scatter(centers_p[:, 0], centers_p[:, 1], c='black', s=200, marker='*', edgecolors='white', linewidths=1)\n"
        "    ax_p.set_title(f'K-Means (k={k_p}) — PCA Projection', fontsize=12, fontweight='bold')\n"
        "    ax_p.set_xlabel('Principal Component 1', fontsize=10)\n"
        "    ax_p.set_ylabel('Principal Component 2', fontsize=10)\n"
        "plt.tight_layout()\n"
        "plt.savefig('../reports/figures/pca_clusters.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('PCA cluster plot saved.')"
    ),

]

for i, cell in enumerate(day3_cells):
    nb["cells"].insert(insert_before + i, cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Injected {len(day3_cells)} Day 3 cells into notebook.")
