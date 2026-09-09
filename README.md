# Kolam-Design-Geometry-Analysis-
An interactive data science workbench and computational geometry framework that analyzes traditional Kolam patterns using unsupervised machine learning. By extracting structural features—such as rotational symmetry, loop connectivity, and stroke density—this project leverages Principal Component Analysis (PCA) and t-SNE to automatically discover, cluster, and map pattern typologies (Sikku, radial grid, and line designs) for cultural heritage preservation.
# Key Features
Image Preprocessing & Binarization: Automatically converts raw Kolam photographs and drawings into high-contrast binary masks using adaptive and Otsu thresholding.

Geometric Feature Extraction: Quantifies key structural properties:

Rotational and bilateral symmetry indices (via IoU matrix transformations).

Topological descriptors (Euler number, loop counts, contour aspect ratios, Hu moments).

Radial pixel distribution and stroke density around grid centers (pulli).

Unsupervised Clustering: Uses PCA for variance driver identification and t-SNE for 2D/3D cluster mapping of pattern families.

Interactive Operations Dashboard: Built with Streamlit/Dash featuring a dark slate (#1E293B) and terracotta (#E07A5F) palette, allowing judges and researchers to explore scatter plots and inspect feature vectors in real time.
# Project Architecture 
kolam-geometry-analysis/
├── data/
│   ├── raw_images/          # Input Kolam dataset
│   └── processed_masks/     # Binarized & segmented masks
├── src/
│   ├── preprocessing.py     # Grayscale, thresholding, & contour extraction
│   ├── feature_extractor.py # Geometric symmetry & topology vector generation
│   ├── clustering.py        # PCA & t-SNE dimensionality reduction
│   └── dashboard.py         # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md
# Tech Stack
Language: Python 3.10+

Computer Vision: OpenCV, NumPy

Machine Learning: scikit-learn (PCA, t-SNE, StandardScaler)

Data & Visualization: Pandas, Matplotlib, Seaborn, Plotly

Dashboard: Streamlit
