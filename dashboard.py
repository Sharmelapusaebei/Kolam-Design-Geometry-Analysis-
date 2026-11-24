import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering import KolamClusterEngine
from feature_extractor import KolamFeatureExtractor

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
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕸️ Kolam Design Geometry Analysis")
st.markdown(
    "Unsupervised Machine Learning Workbench for Structural Pattern Family Discovery"
)

# Sidebar controls
st.sidebar.header("Navigation & Settings")
projection_method = st.sidebar.radio(
    "Dimensionality Reduction Method", ["t-SNE", "PCA"]
)
num_clusters = st.sidebar.slider("Number of Pattern Typologies", 2, 5, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Kolam Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload a Kolam image",
    type=["png", "jpg", "jpeg", "webp"],
)


@st.cache_data
def generate_sample_dataset(n_samples=20):
    np.random.seed(42)
    rows = []
    for i in range(n_samples):
        rows.append(
            {
                "image_name": f"kolam_sample_{i+1}.png",
                "vertical_symmetry": float(np.random.uniform(0.6, 0.99)),
                "horizontal_symmetry": float(np.random.uniform(0.5, 0.98)),
                "rotational_symmetry_90": float(np.random.uniform(0.4, 0.95)),
                "loop_count": int(np.random.randint(1, 15)),
                "stroke_density": float(np.random.uniform(0.1, 0.45)),
                "hu_moment_1": float(np.random.uniform(1.0, 3.5)),
                "hu_moment_2": float(np.random.uniform(0.1, 1.2)),
            }
        )
    return pd.DataFrame(rows)


def extract_uploaded_features(file_bytes, filename, corpus_df):
    extractor = KolamFeatureExtractor()
    raw_feats = extractor.extract_from_bytes(file_bytes)
    row = {
        col: raw_feats.get(col, float(corpus_df[col].median()))
        for col in FEATURE_COLS
    }
    row["image_name"] = filename
    return row


def build_archetype_summaries(df, x_col, y_col):
    summaries = []
    overall_means = df[FEATURE_COLS].mean(numeric_only=True)

    for cluster_val, group in df.groupby("cluster"):
        cx = group[x_col].mean()
        cy = group[y_col].mean()
        dists = np.hypot(group[x_col] - cx, group[y_col] - cy)
        medoid_row = group.loc[dists.idxmin()]

        group_means = group[FEATURE_COLS].mean(numeric_only=True)
        diffs = (group_means - overall_means) / overall_means.replace(0, 1.0)

        traits = []
        if diffs.get("loop_count", 0) > 0.12:
            traits.append(
                f"Higher contour/loop count (~{group_means['loop_count']:.1f})"
            )
        elif diffs.get("loop_count", 0) < -0.12:
            traits.append(
                f"Lower contour/loop count (~{group_means['loop_count']:.1f})"
            )

        sym_avg = float(
            np.mean(
                [
                    group_means.get("vertical_symmetry", 0.0),
                    group_means.get("horizontal_symmetry", 0.0),
                    group_means.get("rotational_symmetry_90", 0.0),
                ]
            )
        )
        traits.append(f"Mean symmetry score: {sym_avg:.2f}")

        if diffs.get("stroke_density", 0) > 0.12:
            traits.append("Denser foreground stroke coverage")
        elif diffs.get("stroke_density", 0) < -0.12:
            traits.append("Sparser foreground stroke coverage")

        summaries.append(
            {
                "cluster": cluster_val,
                "count": len(group),
                "medoid": medoid_row["image_name"],
                "traits": traits,
            }
        )
    return summaries


def build_markdown_summary(summaries, method_name, metrics):
    lines = [
        "# Kolam Structural Pattern-Family Summary",
        f"- **Projection Method:** {method_name}",
    ]
    if metrics.get("silhouette") is not None:
        lines.append(f"- **Silhouette Score:** {metrics['silhouette']:.3f}")
    if metrics.get("davies_bouldin") is not None:
        lines.append(f"- **Davies–Bouldin Index:** {metrics['davies_bouldin']:.3f}")
    lines.append("")

    for item in summaries:
        lines.append(f"## {item['cluster']} (n = {item['count']})")
        lines.append(f"- **Representative / Medoid:** `{item['medoid']}`")
        for t in item["traits"]:
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines)


df_features = generate_sample_dataset()
x_col = "tsne_x" if projection_method == "t-SNE" else "pca_x"
y_col = "tsne_y" if projection_method == "t-SNE" else "pca_y"

uploaded_row = None
upload_error = None
if uploaded_file is not None:
    try:
        uploaded_row = extract_uploaded_features(
            uploaded_file.getvalue(), uploaded_file.name, df_features
        )
        df_working = pd.concat(
            [df_features, pd.DataFrame([uploaded_row])], ignore_index=True
        )
    except Exception as exc:
        upload_error = str(exc)
        df_working = df_features.copy()
else:
    df_working = df_features.copy()

engine = KolamClusterEngine(n_clusters=num_clusters)
df_results = engine.fit_transform(df_working)
metrics = engine.get_validation_metrics()

if uploaded_row is not None:
    query_mask = df_results["image_name"] == uploaded_row["image_name"]
    query_df = df_results[query_mask]
    corpus_results = df_results[~query_mask].copy()
else:
    query_df = pd.DataFrame()
    corpus_results = df_results.copy()

archetypes = build_archetype_summaries(corpus_results, x_col, y_col)

# Top Validation Bar
m1, m2, m3 = st.columns(3)
m1.metric("Corpus Size", f"{len(corpus_results)} patterns")
m2.metric(
    "Silhouette Score",
    f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "N/A",
)
m3.metric(
    "Davies–Bouldin Index",
    f"{metrics['davies_bouldin']:.3f}"
    if metrics["davies_bouldin"] is not None
    else "N/A",
)

# 1. Cluster Archetype Summary Cards
st.subheader("Cluster Archetype Summary")
card_cols = st.columns(min(len(archetypes), 3) if archetypes else 1)
for idx, item in enumerate(archetypes):
    with card_cols[idx % len(card_cols)]:
        traits_html = "".join(f"<li>{t}</li>" for t in item["traits"])
        st.markdown(
            f"""
            <div class="archetype-card">
                <strong>{item['cluster']}</strong> &nbsp;·&nbsp; <span>{item['count']} patterns</span><br/>
                <small>Medoid: <code>{item['medoid']}</code></small>
                <ul style="margin-top:8px; margin-bottom:0; padding-left:18px;">
                    {traits_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Main Layout: Scatter Plot + Feature Table
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Typology Cluster Mapping ({projection_method})")
    fig = px.scatter(
        corpus_results,
        x=x_col,
        y=y_col,
        color="cluster",
        hover_data=["image_name", "loop_count", "vertical_symmetry"],
        color_discrete_sequence=["#E07A5F", "#3D405B", "#81B29A", "#F28F3B", "#6B705C"],
        title=f"2D Feature Projection via {projection_method}",
    )

    if not query_df.empty:
        q_row = query_df.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[q_row[x_col]],
                y=[q_row[y_col]],
                mode="markers+text",
                name="Uploaded Kolam",
                text=[f"Query: {q_row['image_name']}"],
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
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Feature Vectors")
    st.dataframe(
        df_results[
            ["image_name", "cluster", "vertical_symmetry", "loop_count"]
        ],
        hide_index=True,
    )

# 2. Live Custom Kolam Inspection Panel
if upload_error:
    st.warning(f"Could not process uploaded image: {upload_error}")
elif uploaded_file is not None and not query_df.empty:
    st.markdown("---")
    st.subheader("Live Uploaded Kolam Projection")
    u_col1, u_col2 = st.columns([1, 2])
    with u_col1:
        st.image(
            io.BytesIO(uploaded_file.getvalue()),
            caption=uploaded_file.name,
            use_container_width=True,
        )
    with u_col2:
        q_info = query_df.iloc[0]
        st.markdown(f"**Assigned Typology:** `{q_info['cluster']}`")
        st.dataframe(
            pd.DataFrame([uploaded_row])[FEATURE_COLS].round(4),
            hide_index=True,
        )

# 3. One-Click Export Bar
st.markdown("---")
st.subheader("One-Click Export for Reports & Slides")
exp1, exp2, exp3 = st.columns(3)

with exp1:
    st.download_button(
        label="Download Feature & Cluster CSV",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name=f"kolam_clusters_{projection_method.lower()}.csv",
        mime="text/csv",
    )

with exp2:
    md_report = build_markdown_summary(archetypes, projection_method, metrics)
    st.download_button(
        label="Download Archetype Summary (MD)",
        data=md_report.encode("utf-8"),
        file_name=f"kolam_archetypes_{projection_method.lower()}.md",
        mime="text/markdown",
    )

with exp3:
    plot_html = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label="Download Projection Plot (HTML)",
        data=plot_html,
        file_name=f"kolam_projection_{projection_method.lower()}.html",
        mime="text/html",
    )

st.markdown("---")
st.subheader("Extracted Metrics Breakdown")
st.line_chart(
    corpus_results.set_index("image_name")[
        ["vertical_symmetry", "horizontal_symmetry", "rotational_symmetry_90"]
    ]
)
