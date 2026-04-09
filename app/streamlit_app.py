"""
streamlit_app.py
----------------
Driver Behavior Clustering — Interactive Analysis Dashboard

Sections:
  1. Project Overview         — business context and key metrics
  2. Cluster Explorer         — interactive PCA scatter (Plotly)
  3. Algorithm Comparison     — K-Means vs DBSCAN vs Agglomerative table
  4. Driver Segment Profiles  — radar chart and pricing personas
  5. Predict New Driver       — live cluster assignment from sidebar inputs
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ------------------------------------------------------------------ #
# Path setup                                                          #
# ------------------------------------------------------------------ #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ------------------------------------------------------------------ #
# Page configuration                                                  #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Driver Behavior Clustering",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Load config                                                         #
# ------------------------------------------------------------------ #
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ------------------------------------------------------------------ #
# Load data                                                           #
# ------------------------------------------------------------------ #
DATA_PATH = os.path.join(ROOT, "data", "driver_behavior.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def run_kmeans(k: int):
    df = load_data()
    km = KMeans(
        n_clusters=k,
        random_state=cfg["clustering"]["random_state"],
        n_init=cfg["clustering"]["n_init"],
    )
    labels = km.fit_predict(df)
    return df, km, labels

@st.cache_data
def run_pca(k: int):
    df, km, labels = run_kmeans(k)
    pca = PCA(n_components=2, random_state=cfg["clustering"]["random_state"])
    coords = pca.fit_transform(df)
    centers_2d = pca.transform(km.cluster_centers_)
    result = df.copy()
    result["PC1"] = coords[:, 0]
    result["PC2"] = coords[:, 1]
    result["Cluster"] = labels.astype(str)
    return result, centers_2d, pca.explained_variance_ratio_

@st.cache_data
def compute_algo_comparison():
    df = load_data()
    X = df.values

    results = []

    # K-Means k=2
    km = KMeans(n_clusters=2, random_state=cfg["clustering"]["random_state"], n_init=cfg["clustering"]["n_init"])
    lbl = km.fit_predict(df)
    results.append({
        "Algorithm": "K-Means (k=2)",
        "Clusters Found": 2,
        "Silhouette Score": round(silhouette_score(X, lbl), 4),
        "Davies-Bouldin Index": round(davies_bouldin_score(X, lbl), 4),
        "Calinski-Harabasz Index": round(calinski_harabasz_score(X, lbl), 4),
    })

    # DBSCAN best config
    best_sil = -99
    best_row = None
    for eps in cfg["dbscan"]["eps_values"]:
        for ms in cfg["dbscan"]["min_samples_values"]:
            db = DBSCAN(eps=eps, min_samples=ms)
            dl = db.fit_predict(df)
            mask = dl != -1
            n_unique = len(set(dl[mask]))
            if n_unique < 2:
                continue
            sil = round(silhouette_score(X[mask], dl[mask]), 4)
            if sil > best_sil:
                best_sil = sil
                best_row = {
                    "Algorithm": f"DBSCAN (eps={eps}, min_samples={ms})",
                    "Clusters Found": n_unique,
                    "Silhouette Score": sil,
                    "Davies-Bouldin Index": round(davies_bouldin_score(X[mask], dl[mask]), 4),
                    "Calinski-Harabasz Index": round(calinski_harabasz_score(X[mask], dl[mask]), 4),
                }
    if best_row:
        results.append(best_row)

    # Agglomerative k=2
    agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    al = agg.fit_predict(df)
    results.append({
        "Algorithm": "Agglomerative Hierarchical (k=2)",
        "Clusters Found": 2,
        "Silhouette Score": round(silhouette_score(X, al), 4),
        "Davies-Bouldin Index": round(davies_bouldin_score(X, al), 4),
        "Calinski-Harabasz Index": round(calinski_harabasz_score(X, al), 4),
    })

    return pd.DataFrame(results)


# ------------------------------------------------------------------ #
# Persona definitions                                                 #
# ------------------------------------------------------------------ #
PERSONAS = {
    0: {
        "label": "Cautious Commuters",
        "risk_level": "Low",
        "key_traits": "Low braking events, short trips, daytime driving",
        "pricing_action": "Offer 10–20% loyalty discount; upsell telematics programme",
        "color": "#2563EB",
    },
    1: {
        "label": "High-Exposure Drivers",
        "risk_level": "Elevated",
        "key_traits": "Elevated speed variance, night driving, frequent hard braking",
        "pricing_action": "Apply risk-adjusted surcharge; offer telematics opt-in for premium reduction",
        "color": "#DC2626",
    },
}

# ------------------------------------------------------------------ #
# Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("Driver Behavior Clustering")
    st.markdown("**IBM Portfolio Project**")
    st.markdown("Unsupervised segmentation of 10,000 insurance policyholders by driving behaviour.")
    st.divider()

    section = st.radio(
        "Navigate to",
        [
            "Project Overview",
            "Cluster Explorer",
            "Algorithm Comparison",
            "Segment Profiles",
            "Predict New Driver",
        ],
    )

    st.divider()
    k_select = st.slider("Number of clusters (k)", min_value=2, max_value=5, value=2, step=1,
                         help="Controls the K-Means solution shown in the Cluster Explorer.")

# ------------------------------------------------------------------ #
# Section 1: Project Overview                                         #
# ------------------------------------------------------------------ #
if section == "Project Overview":
    st.title("Driver Behavior Clustering")
    st.markdown(
        "An auto-insurance company is redesigning its pricing model using unsupervised "
        "learning to segment policyholders by driving behaviour. This dashboard presents "
        "the full analytical workflow — from algorithm selection to actionable driver personas "
        "and a live prediction tool for new policy applicants."
    )

    st.divider()

    df, km, labels = run_kmeans(cfg["clustering"]["final_k"])
    feature_cols = df.columns.tolist()
    X = df.values
    sil = round(silhouette_score(X, labels), 4)
    dbi = round(davies_bouldin_score(X, labels), 4)
    chi = round(calinski_harabasz_score(X, labels), 4)
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Algorithm", "K-Means")
    col2.metric("Optimal Clusters", cfg["clustering"]["final_k"])
    col3.metric("Silhouette Score", sil, help="Higher is better. Range: -1 to 1.")
    col4.metric("Davies-Bouldin Index", dbi, help="Lower is better.")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Calinski-Harabasz Index", f"{chi:,.0f}", help="Higher is better.")
    col6.metric("Dataset Size", "10,000 drivers")
    col7.metric("Features", len(feature_cols))
    for c_id, count in cluster_counts.items():
        col8.metric(f"Cluster {c_id} — {PERSONAS.get(c_id, {}).get('label', str(c_id))}", f"{count:,} drivers")

    st.divider()
    st.subheader("Algorithms Evaluated")
    st.markdown(
        "Three clustering paradigms were compared on the same validation metrics. "
        "K-Means was selected for its interpretable centroids, fixed segment count, "
        "and native out-of-sample prediction support."
    )
    comparison_df = compute_algo_comparison()
    st.dataframe(comparison_df, use_container_width=True)

# ------------------------------------------------------------------ #
# Section 2: Cluster Explorer                                         #
# ------------------------------------------------------------------ #
elif section == "Cluster Explorer":
    st.title("Cluster Explorer")
    st.markdown(
        f"Interactive PCA projection of the K-Means solution with **k={k_select}**. "
        "Each point represents one driver. Hover to inspect feature values. "
        "Star markers indicate cluster centroids."
    )

    pca_df, centers_2d, var_ratio = run_pca(k_select)

    # Map cluster IDs to persona labels where available
    pca_df["Segment"] = pca_df["Cluster"].apply(
        lambda c: PERSONAS.get(int(c), {}).get("label", f"Cluster {c}")
    )

    feature_cols = [c for c in pca_df.columns if c not in ["PC1", "PC2", "Cluster", "Segment"]]
    hover_data = {col: True for col in feature_cols}
    hover_data["PC1"] = False
    hover_data["PC2"] = False

    color_map = {PERSONAS[i]["label"]: PERSONAS[i]["color"] for i in PERSONAS}

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Segment",
        color_discrete_map=color_map,
        hover_data=hover_data,
        opacity=0.65,
        title=f"K-Means Clustering — PCA Projection (k={k_select})",
        labels={
            "PC1": f"PC1 ({var_ratio[0]*100:.1f}% variance)",
            "PC2": f"PC2 ({var_ratio[1]*100:.1f}% variance)",
        },
    )

    # Centroids
    for i, center in enumerate(centers_2d):
        label = PERSONAS.get(i, {}).get("label", f"Cluster {i}")
        fig.add_scatter(
            x=[center[0]],
            y=[center[1]],
            mode="markers",
            marker=dict(symbol="star", size=18, color="black", line=dict(color="white", width=1)),
            name=f"Centroid {i}",
            showlegend=True,
        )

    fig.update_layout(
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"PCA retains {(var_ratio[0]+var_ratio[1])*100:.1f}% of total variance in this 2D projection. "
        "The full clustering was performed on all 5 original features."
    )

# ------------------------------------------------------------------ #
# Section 3: Algorithm Comparison                                     #
# ------------------------------------------------------------------ #
elif section == "Algorithm Comparison":
    st.title("Algorithm Comparison")
    st.markdown(
        "Three clustering approaches were evaluated. The table below shows each "
        "algorithm's best-performing configuration on this dataset."
    )

    comparison_df = compute_algo_comparison()

    # Highlight best values in each metric column
    def highlight_best(col):
        if col.name == "Silhouette Score":
            best = col.max()
            return ["background-color: #d1fae5; font-weight: bold" if v == best else "" for v in col]
        elif col.name == "Davies-Bouldin Index":
            best = col.min()
            return ["background-color: #d1fae5; font-weight: bold" if v == best else "" for v in col]
        elif col.name == "Calinski-Harabasz Index":
            best = col.max()
            return ["background-color: #d1fae5; font-weight: bold" if v == best else "" for v in col]
        return [""] * len(col)

    st.dataframe(
        comparison_df.style.apply(highlight_best),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Why K-Means Was Selected")
    st.markdown(
        """
**K-Means (k=2)** achieved the highest Silhouette Score and lowest Davies-Bouldin Index, confirming 
the most compact and well-separated clusters. Beyond metric performance, K-Means was selected for 
three business-specific reasons:

1. **Interpretable centroids.** Each cluster has a concrete feature profile that maps directly 
   to a driver persona the pricing team can act on. DBSCAN and agglomerative methods do not 
   naturally provide this.

2. **Fixed, predictable segment count.** The business requirement is separate pricing models 
   per group. K-Means delivers a deterministic number of segments, which is operationally 
   necessary.

3. **Production compatibility.** K-Means natively supports out-of-sample prediction via 
   `.predict()`, enabling the model to score new drivers at policy inception without retraining.

**When DBSCAN is the right choice:** DBSCAN is superior when the number of clusters is unknown, 
cluster shapes are non-convex, or noise detection (anomaly/fraud identification) is the primary 
objective. None of those conditions apply here.
        """
    )

# ------------------------------------------------------------------ #
# Section 4: Segment Profiles                                         #
# ------------------------------------------------------------------ #
elif section == "Segment Profiles":
    st.title("Driver Segment Profiles")
    st.markdown("Each cluster is profiled by its mean feature values and assigned a business label.")

    df, km, labels = run_kmeans(cfg["clustering"]["final_k"])
    feature_cols = df.columns.tolist()
    df_with_labels = df.copy()
    df_with_labels["Cluster"] = labels
    profiles = df_with_labels.groupby("Cluster")[feature_cols].mean()
    cluster_sizes = df_with_labels["Cluster"].value_counts().sort_index()

    # Persona cards
    cols = st.columns(len(PERSONAS))
    for i, (cluster_id, persona) in enumerate(PERSONAS.items()):
        with cols[i]:
            st.markdown(f"### Cluster {cluster_id}")
            st.markdown(f"**{persona['label']}**")
            st.markdown(f"**Risk level:** {persona['risk_level']}")
            st.markdown(f"**Drivers:** {cluster_sizes.get(cluster_id, 0):,}")
            st.markdown(f"**Key traits:** {persona['key_traits']}")
            st.markdown(f"**Pricing action:** {persona['pricing_action']}")

    st.divider()

    # Radar chart
    st.subheader("Behavioural Profile — Radar Chart")
    N = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    norm_profiles = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)

    fig_r = go.Figure()
    for cluster_id in profiles.index:
        persona = PERSONAS.get(cluster_id, {"label": str(cluster_id), "color": "gray"})
        values = norm_profiles.loc[cluster_id].tolist()
        values_closed = values + values[:1]
        fig_r.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=feature_cols + [feature_cols[0]],
            fill="toself",
            name=f"Cluster {cluster_id} — {persona['label']}",
            line=dict(color=persona["color"], width=2),
            fillcolor=persona["color"],
            opacity=0.3,
        ))

    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Driver Segment Behavioural Profiles (Normalised)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=500,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    # Heatmap
    st.subheader("Centroid Feature Heatmap")
    fig_h = px.imshow(
        profiles.round(3),
        color_continuous_scale="RdBu_r",
        text_auto=True,
        aspect="auto",
        title="Mean Feature Value per Cluster",
        labels=dict(color="Mean Value"),
    )
    fig_h.update_layout(height=300)
    st.plotly_chart(fig_h, use_container_width=True)

    st.divider()

    # Pricing table
    st.subheader("Pricing Implications")
    pricing_rows = [
        {
            "Segment": cluster_id,
            "Label": PERSONAS[cluster_id]["label"],
            "Risk Level": PERSONAS[cluster_id]["risk_level"],
            "Key Traits": PERSONAS[cluster_id]["key_traits"],
            "Recommended Pricing Action": PERSONAS[cluster_id]["pricing_action"],
        }
        for cluster_id in PERSONAS
    ]
    st.dataframe(pd.DataFrame(pricing_rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Section 5: Predict New Driver                                       #
# ------------------------------------------------------------------ #
elif section == "Predict New Driver":
    st.title("Predict New Driver Segment")
    st.markdown(
        "Enter a new driver's feature values using the sliders below. "
        "The model will assign them to the most appropriate cluster and return "
        "the segment label, risk level, and recommended pricing action."
    )

    df_ref = load_data()
    feature_cols = df_ref.columns.tolist()

    # Load or train the model
    MODEL_PATH = os.path.join(ROOT, "reports", "kmeans_k2_final.joblib")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        df, km, labels = run_kmeans(cfg["clustering"]["final_k"])
        model = km

    st.subheader("Driver Feature Inputs")
    input_vals = {}

    slider_cols = st.columns(len(feature_cols))
    for col, feat in zip(slider_cols, feature_cols):
        f_min = float(df_ref[feat].min())
        f_max = float(df_ref[feat].max())
        f_mean = float(df_ref[feat].mean())
        with col:
            input_vals[feat] = st.slider(
                feat,
                min_value=round(f_min, 3),
                max_value=round(f_max, 3),
                value=round(f_mean, 3),
                step=round((f_max - f_min) / 100, 4),
                format="%.3f",
            )

    if st.button("Predict Segment", type="primary"):
        new_driver = pd.DataFrame([input_vals])
        predicted_cluster = int(model.predict(new_driver)[0])
        persona = PERSONAS.get(predicted_cluster, {"label": str(predicted_cluster), "risk_level": "Unknown", "pricing_action": "—"})

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Assigned Cluster", f"Cluster {predicted_cluster}")
        r2.metric("Segment Label", persona["label"])
        r3.metric("Risk Level", persona["risk_level"])

        st.info(f"**Recommended Pricing Action:** {persona['pricing_action']}")

        st.markdown("**Input feature summary:**")
        st.dataframe(new_driver, use_container_width=True, hide_index=True)

        st.caption(
            "This prediction uses the K-Means model trained on 10,000 historical policyholders. "
            "Final pricing decisions must incorporate actuarial review and comply with applicable "
            "regulatory requirements."
        )
