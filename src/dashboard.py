import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from clustering import KolamClusterEngine
from feature_extractor import KolamFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="Kolam Design Geometry Workbench",
    page_icon="🕸️",
    layout="wide",
)

# Custom Dark Slate & Terracotta Theme CSS
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
    .stButton>button {
        background-color: #E07A5F;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #F28F3B;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🕸️ Kolam Design Geometry Analysis")
st.markdown(
    "Unsupervised Machine Learning Workbench for Structural Pattern Family Discovery"
)

# Sidebar
st.sidebar.header("Navigation & Settings")
projection_method = st.sidebar.radio(
    "Dimensionality Reduction Method", ["t-SNE", "PCA"]
)
num_clusters = st.sidebar.slider("Number of Pattern Typologies", 2, 5, 3)

# Dummy dataset generator for initial demonstration
@st.cache_data
def generate_sample_dataset(n_samples=20):
    np.random.seed(42)
    data = []
    for i in range(n_samples):
        data.append(
            {
                "image_name": f"kolam_sample_{i+1}.png",
                "vertical_symmetry": np.random.uniform(0.6, 0.99),
                "horizontal_symmetry": np.random.uniform(0.5, 0.98),
                "rotational_symmetry_90": np.random.uniform(0.4, 0.95),
                "loop_count": np.random.randint(1, 15),
                "stroke_density": np.random.uniform(0.1, 0.45),
                "hu_moment_1": np.random.uniform(1.0, 3.5),
                "hu_moment_2": np.random.uniform(0.1, 1.2),
            }
        )
    return pd.DataFrame(data)


df_features = generate_sample_dataset()

# Execute Clustering Pipeline
engine = KolamClusterEngine(n_clusters=num_clusters)
df_results = engine.fit_transform(df_features)

x_col = "tsne_x" if projection_method == "t-SNE" else "pca_x"
y_col = "tsne_y" if projection_method == "t-SNE" else "pca_y"

# Main Layout: Two Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Typology Cluster Mapping ({projection_method})")
    fig = px.scatter(
        df_results,
        x=x_col,
        y=y_col,
        color="cluster",
        hover_data=["image_name", "loop_count", "vertical_symmetry"],
        color_discrete_sequence=["#E07A5F", "#3D405B", "#81B29A", "#F28F3B"],
        title=f"2D Feature Projection via {projection_method}",
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

st.markdown("---")
st.subheader("Extracted Metrics Breakdown")
st.line_chart(
    df_results.set_index("image_name")[
        ["vertical_symmetry", "horizontal_symmetry", "rotational_symmetry_90"]
    ]
)
