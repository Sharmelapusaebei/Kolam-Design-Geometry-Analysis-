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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
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
    .top-header {
        text-align: center;
        padding-top: 6px;
        padding-bottom: 10px;
    }
    .top-header h1 {
        text-align: center !important;
        margin-bottom: 6px;
    }
    .top-subtitle {
        text-align: center;
        color: #CBD5E1;
        font-size: 1rem;
        margin-bottom: 8px;
    }
    .projection-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 18px;
        margin-top: 8px;
        margin-bottom: 16px;
        text-align: center;
    }
    .projection-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #E07A5F;
        margin-bottom: 4px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-left: 5px solid #E07A5F;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }
    .family-card-1 {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-top: 4px solid #E07A5F;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .family-card-2 {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-top: 4px solid #81B29A;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .step-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
    }
    .badge-f1 {
        background-color: rgba(224, 122, 95, 0.2);
        color: #E07A5F;
        border: 1px solid #E07A5F;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-f2 {
        background-color: rgba(129, 178, 154, 0.2);
        color: #81B29A;
        border: 1px solid #81B29A;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def generate_two_family_dataset(n_per_family=25):
    """Generates a balanced 2-family Kolam feature dataset (N = 50 by default)."""
    rng = np.random.default_rng(42)
    rows = []

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


def render_archetype_preview(family_name, size=180):
    """Generates a visual preview thumbnail representing each structural family's geometry."""
    canvas = np.full((size, size, 3), [15, 23, 42], dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    if family_name == "Family 1":
        color = np.array([224, 122, 95], dtype=np.uint8)
        for dx in [-42, 0, 42]:
            for dy in [-42, 0, 42]:
                dist = np.hypot(x - (cx + dx), y - (cy + dy))
                ring = (dist >= 16) & (dist <= 20)
                dot = dist <= 3
                canvas[ring | dot] = color
        outer = np.abs(np.abs(x - cx) + np.abs(y - cy) - 72) <= 2
        canvas[outer] = color
    else:
        color = np.array([129, 178, 154], dtype=np.uint8)
        dist = np.hypot(x - cx, y - cy)
        angle = np.arctan2(y - cy, x - cx)
        petal = np.abs(dist - (46 + 18 * np.cos(4 * angle))) <= 4
        inner = (dist >= 14) & (dist <= 22)
        core = dist <= 6
        canvas[petal | inner | core] = color

    return canvas


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
            traits.append(f"Higher loop/contour count (~{group_means['loop_count']:.1f} loops)")
        elif diffs.get("loop_count", 0) < -0.08:
            traits.append(f"Lower loop/contour count (~{group_means['loop_count']:.1f} loops)")
        else:
            traits.append(f"Moderate loop/contour count (~{group_means['loop_count']:.1f} loops)")

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
                "means": group_means,
                "members": group["image_name"].tolist(),
            }
        )
    return summaries


def build_projection_figure(
    corpus_df, x_col, y_col, method_name, highlight_row=None, highlight_label="Query"
):
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


st.markdown(
    """
    <div class="top-header">
        <h1>🕸️ Kolam Design Geometry Analysis &amp; Structural Pattern-Family Discovery Workbench</h1>
        <div class="top-subtitle">
            Unsupervised Geometric &amp; Topological Machine Learning System for Two-Family Kolam Classification
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df_corpus = generate_two_family_dataset(n_per_family=25)

tab_home, tab_map, tab_upload, tab_sim, tab_user = st.tabs(
    [
        "🏠 Home",
        "🗺️ Cluster Map",
        "📤 Upload & Classify Family",
        "⚡ Live Simulation",
        "👤 User & Exports",
    ]
)

with tab_home:
    _, proj_mid, _ = st.columns([1, 1.6, 1])
    with proj_mid:
        st.markdown(
            """
            <div class="projection-box">
                <div class="projection-title">Projection Settings · Two-Family Mode (Family 1 &amp; Family 2)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        projection_method = st.radio(
            "Dimensionality Reduction Method",
            ["t-SNE", "PCA"],
            horizontal=True,
            key="home_projection_method",
        )

    x_col = "tsne_x" if projection_method == "t-SNE" else "pca_x"
    y_col = "tsne_y" if projection_method == "t-SNE" else "pca_y"

    base_engine = KolamClusterEngine(n_clusters=NUM_FAMILIES)
    base_results = normalize_family_labels(base_engine.fit_transform(df_corpus))
    metrics = base_engine.get_validation_metrics()
    archetypes = build_archetype_summaries(base_results, x_col, y_col)
    base_fig = build_projection_figure(base_results, x_col, y_col, projection_method)

    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin-top:0; margin-bottom:6px;">Discovering Structural DNA in Traditional Kolam Art</h3>
            <p style="margin-bottom:0; color:#CBD5E1;">
                Instead of treating Kolam designs as flat pixels or relying on manual labels, this workbench extracts
                geometric symmetry, enclosed loop topology, stroke density, and Hu shape moments to separate patterns
                into <strong>2 distinct structural families</strong> (<strong>Family 1</strong> and <strong>Family 2</strong>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("---")

    st.subheader("Family 1 vs. Family 2: What Each Design Family Represents")
    arch_cols = st.columns(2)
    for idx, item in enumerate(archetypes):
        fam = item["family"]
        card_class = "family-card-1" if fam == "Family 1" else "family-card-2"
        badge_class = "badge-f1" if fam == "Family 1" else "badge-f2"
        subtitle = (
            "Multi-Loop & High-Symmetry Weave Designs"
            if fam == "Family 1"
            else "Dense-Stroke & Compact Geometric Motifs"
        )
        traits_html = "".join(f"<li>{t}</li>" for t in item["traits"])

        with arch_cols[idx % 2]:
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            img_col, desc_col = st.columns([1, 2])
            with img_col:
                st.image(
                    render_archetype_preview(fam),
                    caption=f"Medoid: {item['medoid']}",
                    use_container_width=True,
                )
            with desc_col:
                st.markdown(
                    f"""
                    <span class="{badge_class}">{fam} · {item['count']} patterns</span>
                    <h4 style="margin-top:8px; margin-bottom:6px; color:#F8FAFC;">{subtitle}</h4>
                    <ul style="margin-top:4px; margin-bottom:0; padding-left:18px; color:#CBD5E1;">
                        {traits_html}
                    </ul>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("How the System Identifies Family 1 vs. Family 2")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            """
            <div class="step-card">
                <strong>Step 1 · Binarization</strong><br/>
                <small style="color:#CBD5E1;">
                Converts every Kolam image to grayscale, resizes to 256×256, and isolates foreground strokes via Otsu thresholding.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="step-card">
                <strong>Step 2 · Geometry Vector</strong><br/>
                <small style="color:#CBD5E1;">
                Measures vertical, horizontal, and 90° rotational symmetry (IoU), contour loop count, stroke density, and Hu moments.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="step-card">
                <strong>Step 3 · Z-Score &amp; Manifold</strong><br/>
                <small style="color:#CBD5E1;">
                Standardizes all features equally and projects them via PCA (linear) and t-SNE (non-linear) to reveal natural separation.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
            <div class="step-card">
                <strong>Step 4 · 2-Family Assignment</strong><br/>
                <small style="color:#CBD5E1;">
                Groups patterns into <strong>Family 1</strong> or <strong>Family 2</strong> based on structural similarity—no manual labels required.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    comp_col, mem_col = st.columns([1.1, 1.3])

    with comp_col:
        st.subheader("Feature Drivers Separating the Two Families")
        if len(archetypes) == 2:
            f1_means = archetypes[0]["means"]
            f2_means = archetypes[1]["means"]
            comp_df = pd.DataFrame(
                {
                    "Feature": [
                        "Vertical Symmetry",
                        "Horizontal Symmetry",
                        "90° Rotational Symmetry",
                        "Stroke Density",
                        "Normalized Loop Count",
                    ],
                    "Family 1": [
                        f1_means["vertical_symmetry"],
                        f1_means["horizontal_symmetry"],
                        f1_means["rotational_symmetry_90"],
                        f1_means["stroke_density"],
                        f1_means["loop_count"] / 20.0,
                    ],
                    "Family 2": [
                        f2_means["vertical_symmetry"],
                        f2_means["horizontal_symmetry"],
                        f2_means["rotational_symmetry_90"],
                        f2_means["stroke_density"],
                        f2_means["loop_count"] / 20.0,
                    ],
                }
            )
            comp_long = comp_df.melt(
                id_vars="Feature", var_name="Family", value_name="Score"
            )
            bar_fig = px.bar(
                comp_long,
                x="Score",
                y="Feature",
                color="Family",
                barmode="group",
                orientation="h",
                color_discrete_map={"Family 1": "#E07A5F", "Family 2": "#81B29A"},
            )
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0F172A",
                font_color="#F8FAFC",
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    with mem_col:
        st.subheader("Which Images Belong to Which Family?")
        family_filter = st.radio(
            "Filter Corpus Images by Assigned Family",
            ["All (50)", "Family 1", "Family 2"],
            horizontal=True,
        )
        filtered_df = base_results.copy()
        if family_filter != "All (50)":
            filtered_df = filtered_df[filtered_df["family"] == family_filter]

        st.dataframe(
            filtered_df[
                [
                    "image_name",
                    "family",
                    "loop_count",
                    "stroke_density",
                    "vertical_symmetry",
                    "rotational_symmetry_90",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            height=295,
        )

if "home_projection_method" not in st.session_state:
    st.session_state["home_projection_method"] = "t-SNE"
active_projection = st.session_state["home_projection_method"]
x_col = "tsne_x" if active_projection == "t-SNE" else "pca_x"
y_col = "tsne_y" if active_projection == "t-SNE" else "pca_y"

_engine = KolamClusterEngine(n_clusters=NUM_FAMILIES)
_results = normalize_family_labels(_engine.fit_transform(df_corpus))
_metrics = _engine.get_validation_metrics()
_archetypes = build_archetype_summaries(_results, x_col, y_col)
_fig = build_projection_figure(_results, x_col, y_col, active_projection)

with tab_map:
    st.subheader(f"Two-Family Projection ({active_projection})")
    st.plotly_chart(_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Symmetry Metrics Across Corpus Patterns")
    st.line_chart(
        _results.set_index("image_name")[
            ["vertical_symmetry", "horizontal_symmetry", "rotational_symmetry_90"]
        ]
    )

with tab_upload:
    st.subheader("Upload a Kolam Design to Identify Its Family")
    st.markdown(
        "Upload any Kolam image to extract geometric and topological features and determine whether it belongs to **Family 1** or **Family 2**."
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
                active_projection,
                highlight_row=q_match,
                highlight_label="Uploaded Kolam",
            )
            st.plotly_chart(upload_fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to classify uploaded Kolam: {exc}")

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
        active_projection,
        highlight_row=sim_point,
        highlight_label="Simulated",
    )
    st.plotly_chart(sim_fig, use_container_width=True)

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
            data=_results.to_csv(index=False).encode("utf-8"),
            file_name=f"kolam_two_families_{active_projection.lower()}.csv",
            mime="text/csv",
        )

    with exp2:
        md_report = build_markdown_summary(_archetypes, active_projection, _metrics)
        st.download_button(
            label="Download Family Summary (MD)",
            data=md_report.encode("utf-8"),
            file_name=f"kolam_two_families_{active_projection.lower()}.md",
            mime="text/markdown",
        )

    with exp3:
        plot_html = _fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="Download Projection Plot (HTML)",
            data=plot_html,
            file_name=f"kolam_projection_{active_projection.lower()}.html",
            mime="text/html",
        )