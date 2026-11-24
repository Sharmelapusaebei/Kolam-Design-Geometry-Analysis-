import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from clustering import KolamClusterEngine
from feature_extractor import KolamFeatureExtractor

NUM_FAMILIES = 2

FEATURE_COLS = [
    "vertical_symmetry",
    "horizontal_symmetry",
    "rotational_symmetry_90",
    "loop_count",
    "stroke_density",
    "hu_moment_1",
    "hu_moment_2",
]

st.set_page_config(
    page_title="Kolam Design Geometry Workbench",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    .stSidebar {
        background-color: #0F172A !important;
    }
    h1, h2, h3 {
        color: #E07A5F !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #E07A5F;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #F28F3B;
        color: white;
    }
    .archetype-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-left: 4px solid #E07A5F;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕸️ Kolam Design Geometry Analysis")
st.markdown(
    "Unsupervised Machine Learning Workbench for Two-Family Structural Pattern Discovery"
)

# Sidebar controls (locked to 2 pattern families)
st.sidebar.header("Projection Settings")
projection_method = st.sidebar.radio(
    "Dimensionality Reduction Method", ["t-SNE", "PCA"]
)
st.sidebar.info("Classification Mode: **2 Structural Families** (Family 1 & Family 2)")


@st.cache_data
def generate_two_family_dataset(n_per_family=25):
    """Generates a balanced 2-family Kolam feature dataset (N = 50 by default)."""
    rng = np.random.default_rng(42)
    rows = []

    # Family 1: Higher loop complexity & finer/sparser stroke coverage
    for i in range(n_per_family):
        rows.append(
            {
                "image_name": f"kolam_f1_{i+1:02d}.png",
                "seed_family": "Family 1",
                "vertical_symmetry": float(rng.uniform(0.72, 0.99)),
                "horizontal_symmetry": float(rng.uniform(0.68, 0.98)),
                "rotational_symmetry_90": float(rng.uniform(0.65, 0.96)),
                "loop_count": int(rng.integers(8, 18)),
                "stroke_density": float(rng.uniform(0.10, 0.25)),
                "hu_moment_1": float(rng.uniform(1.1, 2.2)),
                "hu_moment_2": float(rng.uniform(0.1, 0.6)),
            }
        )

    # Family 2: Simpler loop structure & denser stroke coverage
    for j in range(n_per_family):
        rows.append(
            {
                "image_name": f"kolam_f2_{j+1:02d}.png",
                "seed_family": "Family 2",
                "vertical_symmetry": float(rng.uniform(0.45, 0.78)),
                "horizontal_symmetry": float(rng.uniform(0.40, 0.75)),
                "rotational_symmetry_90": float(rng.uniform(0.35, 0.72)),
                "loop_count": int(rng.integers(1, 8)),
                "stroke_density": float(rng.uniform(0.26, 0.48)),
                "hu_moment_1": float(rng.uniform(2.3, 3.6)),
                "hu_moment_2": float(rng.uniform(0.65, 1.3)),
            }
        )

    return pd.DataFrame(rows)


def normalize_family_labels(df_in):
    """Maps K-Means cluster labels cleanly to 'Family 1' and 'Family 2'."""
    out = df_in.copy()
    unique_clusters = sorted(out["cluster"].unique())
    mapping = {
        raw_lbl: f"Family {idx + 1}"
        for idx, raw_lbl in enumerate(unique_clusters[:NUM_FAMILIES])
    }
    out["family"] = out["cluster"].map(mapping).fillna("Family 1")
    return out


def extract_uploaded_features(file_bytes, filename, corpus_df):
    extractor = KolamFeatureExtractor()
    raw_feats = extractor.extract_from_bytes(file_bytes)
    row = {
        col: raw_feats.get(col, float(corpus_df[col].median()))
        for col in FEATURE_COLS
    }
    row["image_name"] = filename
    row["seed_family"] = "Uploaded"
    return row


def build_archetype_summaries(df, x_col, y_col):
    summaries = []
    overall_means = df[FEATURE_COLS].mean(numeric_only=True)

    for fam_name in ["Family 1", "Family 2"]:
        group = df[df["family"] == fam_name]
        if group.empty:
            continue

        cx = group[x_col].mean()
        cy = group[y_col].mean()
        dists = np.hypot(group[x_col] - cx, group[y_col] - cy)
        medoid_row = group.loc[dists.idxmin()]

        group_means = group[FEATURE_COLS].mean(numeric_only=True)
        diffs = (group_means - overall_means) / overall_means.replace(0, 1.0)

        traits = []
        if diffs.get("loop_count", 0) > 0.08:
            traits.append(f"Higher loop/contour count (~{group_means['loop_count']:.1f})")
        elif diffs.get("loop_count", 0) < -0.08:
            traits.append(f"Lower loop/contour count (~{group_means['loop_count']:.1f})")
        else:
            traits.append(f"Moderate loop/contour count (~{group_means['loop_count']:.1f})")

        sym_avg = float(
            np.mean(
                [
                    group_means.get("vertical_symmetry", 0.0),
                    group_means.get("horizontal_symmetry", 0.0),
                    group_means.get("rotational_symmetry_90", 0.0),
                ]
            )
        )
        traits.append(f"Mean symmetry index: {sym_avg:.2f}")

        if diffs.get("stroke_density", 0) > 0.08:
            traits.append(f"Denser stroke coverage ({group_means['stroke_density']:.2f})")
        elif diffs.get("stroke_density", 0) < -0.08:
            traits.append(f"Sparser stroke coverage ({group_means['stroke_density']:.2f})")
        else:
            traits.append(f"Balanced stroke coverage ({group_means['stroke_density']:.2f})")

        summaries.append(
            {
                "family": fam_name,
                "count": len(group),
                "medoid": medoid_row["image_name"],
                "traits": traits,
            }
        )
    return summaries


def build_projection_figure(corpus_df, x_col, y_col, method_name, highlight_row=None, highlight_label="Query"):
    fig = px.scatter(
        corpus_df,
        x=x_col,
        y=y_col,
        color="family",
        hover_data=["image_name", "loop_count", "vertical_symmetry", "stroke_density"],
        color_discrete_map={"Family 1": "#E07A5F", "Family 2": "#81B29A"},
        title=f"2D Feature Projection via {method_name} (2 Families)",
    )

    if highlight_row is not None:
        fig.add_trace(
            go.Scatter(
                x=[highlight_row[x_col]],
                y=[highlight_row[y_col]],
                mode="markers+text",
                name=highlight_label,
                text=[f"{highlight_label}: {highlight_row['image_name']}"],
                textposition="top center",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#FDE047",
                    line=dict(color="#0F172A", width=1.5),
                ),
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F172A",
        font_color="#F8FAFC",
    )
    return fig


def build_markdown_summary(summaries, method_name, metrics):
    lines = [
        "# Kolam Two-Family Structural Pattern Summary",
        f"- **Projection Method:** {method_name}",
        "- **Identified Typologies:** 2 (Family 1, Family 2)",
    ]
    if metrics.get("silhouette") is not None:
        lines.append(f"- **Silhouette Score:** {metrics['silhouette']:.3f}")
    if metrics.get("davies_bouldin") is not None:
        lines.append(f"- **Davies–Bouldin Index:** {metrics['davies_bouldin']:.3f}")
    lines.append("")

    for item in summaries:
        lines.append(f"## {item['family']} (n = {item['count']})")
        lines.append(f"- **Representative / Medoid:** `{item['medoid']}`")
        for t in item["traits"]:
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines)


df_corpus = generate_two_family_dataset(n_per_family=25)
x_col = "tsne_x" if projection_method == "t-SNE" else "pca_x"
y_col = "tsne_y" if projection_method == "t-SNE" else "pca_y"

# Base 2-family clustering on the corpus
base_engine = KolamClusterEngine(n_clusters=NUM_FAMILIES)
base_results = normalize_family_labels(base_engine.fit_transform(df_corpus))
metrics = base_engine.get_validation_metrics()
archetypes = build_archetype_summaries(base_results, x_col, y_col)
base_fig = build_projection_figure(base_results, x_col, y_col, projection_method)

# 5 Tabs
tab_overview, tab_map, tab_upload, tab_sim, tab_user = st.tabs(
    [
        "📊 Overview & Archetypes",
        "🗺️ Cluster Map",
        "📤 Upload & Classify Family",
        "⚡ Live Simulation",
        "👤 User & Exports",
    ]
)

# TAB 1: Overview & Archetypes
with tab_overview:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corpus Size", f"{len(base_results)} patterns")
    m2.metric("Design Families", "2 Families")
    m3.metric(
        "Silhouette Score",
        f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A",
    )
    m4.metric(
        "Davies–Bouldin Index",
        f"{metrics['davies_bouldin']:.3f}"
        if metrics["davies_bouldin"] is not None
        else "N/A",
    )

    st.subheader("Two Structural Family Archetypes")
    arch_cols = st.columns(2)
    for idx, item in enumerate(archetypes):
        with arch_cols[idx % 2]:
            traits_html = "".join(f"<li>{t}</li>" for t in item["traits"])
            st.markdown(
                f"""
                <div class="archetype-card">
                    <strong>{item['family']}</strong> &nbsp;·&nbsp; <span>{item['count']} patterns</span><br/>
                    <small>Medoid: <code>{item['medoid']}</code></small>
                    <ul style="margin-top:8px; margin-bottom:0; padding-left:18px;">
                        {traits_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Extracted Feature Table")
    st.dataframe(
        base_results[
            [
                "image_name",
                "family",
                "loop_count",
                "stroke_density",
                "vertical_symmetry",
                "horizontal_symmetry",
                "rotational_symmetry_90",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

# TAB 2: Cluster & Projection Map
with tab_map:
    st.subheader(f"Two-Family Projection ({projection_method})")
    st.plotly_chart(base_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Symmetry Metrics Across Corpus Patterns")
    st.line_chart(
        base_results.set_index("image_name")[
            ["vertical_symmetry", "horizontal_symmetry", "rotational_symmetry_90"]
        ]
    )

# TAB 3: Upload & Classify Family
with tab_upload:
    st.subheader("Upload a Kolam Design to Identify Its Family")
    st.markdown(
        "Upload a Kolam image to extract geometric and topological features and assign it to either **Family 1** or **Family 2**."
    )
    uploaded_file = st.file_uploader(
        "Choose a Kolam image",
        type=["png", "jpg", "jpeg", "webp"],
        key="kolam_family_uploader",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            query_row = extract_uploaded_features(
                file_bytes, uploaded_file.name, df_corpus
            )
            combined_df = pd.concat(
                [df_corpus, pd.DataFrame([query_row])], ignore_index=True
            )
            q_engine = KolamClusterEngine(n_clusters=NUM_FAMILIES)
            q_results = normalize_family_labels(q_engine.fit_transform(combined_df))

            q_match = q_results[q_results["image_name"] == uploaded_file.name].iloc[0]
            q_corpus = q_results[q_results["image_name"] != uploaded_file.name]

            u_left, u_right = st.columns([1, 2])
            with u_left:
                st.image(
                    io.BytesIO(file_bytes),
                    caption=uploaded_file.name,
                    use_container_width=True,
                )
            with u_right:
                st.success(f"Identified Pattern Family: **{q_match['family']}**")
                st.dataframe(
                    pd.DataFrame([query_row])[FEATURE_COLS].round(4),
                    hide_index=True,
                    use_container_width=True,
                )

            upload_fig = build_projection_figure(
                q_corpus,
                x_col,
                y_col,
                projection_method,
                highlight_row=q_match,
                highlight_label="Uploaded Kolam",
            )
            st.plotly_chart(upload_fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to classify uploaded Kolam: {exc}")

# TAB 4: Live Simulation
with tab_sim:
    st.subheader("Live Structural Parameter Simulation")
    st.markdown(
        "Adjust structural parameters below to simulate a Kolam pattern and observe how it shifts between **Family 1** and **Family 2**."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        sim_loops = st.slider("Loop / Contour Count", 1, 20, 10)
        sim_density = st.slider("Stroke Density", 0.05, 0.55, 0.22, step=0.01)
        sim_v_sym = st.slider("Vertical Symmetry", 0.30, 1.00, 0.85, step=0.01)
    with s_col2:
        sim_h_sym = st.slider("Horizontal Symmetry", 0.30, 1.00, 0.80, step=0.01)
        sim_r_sym = st.slider("90° Rotational Symmetry", 0.30, 1.00, 0.78, step=0.01)
        sim_hu1 = st.slider("Hu Moment 1 (log)", 0.8, 4.0, 1.8, step=0.05)

    sim_row = {
        "image_name": "simulated_kolam.png",
        "seed_family": "Simulated",
        "vertical_symmetry": float(sim_v_sym),
        "horizontal_symmetry": float(sim_h_sym),
        "rotational_symmetry_90": float(sim_r_sym),
        "loop_count": int(sim_loops),
        "stroke_density": float(sim_density),
        "hu_moment_1": float(sim_hu1),
        "hu_moment_2": float(df_corpus["hu_moment_2"].median()),
    }

    sim_combined = pd.concat(
        [df_corpus, pd.DataFrame([sim_row])], ignore_index=True
    )
    sim_engine = KolamClusterEngine(n_clusters=NUM_FAMILIES)
    sim_results = normalize_family_labels(sim_engine.fit_transform(sim_combined))

    sim_point = sim_results[sim_results["image_name"] == "simulated_kolam.png"].iloc[0]
    sim_corpus = sim_results[sim_results["image_name"] != "simulated_kolam.png"]

    st.info(f"Simulated Pattern Currently Aligns With: **{sim_point['family']}**")

    sim_fig = build_projection_figure(
        sim_corpus,
        x_col,
        y_col,
        projection_method,
        highlight_row=sim_point,
        highlight_label="Simulated",
    )
    st.plotly_chart(sim_fig, use_container_width=True)

# TAB 5: User & Exports
with tab_user:
    st.subheader("User Workbench Settings")
    u_name = st.text_input("Researcher / Presenter Name", value="Sharmel")
    u_notes = st.text_area(
        "Session Notes",
        value="Two-family unsupervised geometric analysis using PCA and t-SNE.",
    )
    st.caption(f"Active Presenter: **{u_name}** — {u_notes}")

    st.markdown("---")
    st.subheader("One-Click Export for Reports & Slides")
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.download_button(
            label="Download Two-Family CSV",
            data=base_results.to_csv(index=False).encode("utf-8"),
            file_name=f"kolam_two_families_{projection_method.lower()}.csv",
            mime="text/csv",
        )

    with exp2:
        md_report = build_markdown_summary(archetypes, projection_method, metrics)
        st.download_button(
            label="Download Family Summary (MD)",
            data=md_report.encode("utf-8"),
            file_name=f"kolam_two_families_{projection_method.lower()}.md",
            mime="text/markdown",
        )

    with exp3:
        plot_html = base_fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="Download Projection Plot (HTML)",
            data=plot_html,
            file_name=f"kolam_two_families_{projection_method.lower()}.html",
            mime="text/html",
        )