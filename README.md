# 🎯 Enterprise Customer Insights Hub
> **Advanced Unsupervised Machine Learning Pipeline for Behavioral Segmentation & Automated ROI Strategy**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=Plotly&logoColor=white)](https://plotly.com/)

An production-grade, multi-page data analytics platform that ingests raw customer transactional data, engineers high-value behavioral features, isolates statistical variance via IQR, clusters customer cohorts using **K-Means ($k=4$)**, and projects high-dimensional matrices onto a 2D plane via **Principal Component Analysis (PCA)**. 

---

## 🏗️ Core Pipeline Architecture

```mermaid
graph TD
    A[Raw Customer Transactional Data] --> B[Data Cleaning & Missing Value Handling]
    B --> C[IQR Outlier Capping & Standardization]
    C --> D[One-Hot Categorical Encoding]
    D --> E[K-Means Clustering Engine k=4]
    E --> F[PCA Dimensionality Reduction 2D]
    F --> G[Interactive Streamlit Insights Dashboard]
    G --> H1[Executive KPI Overview]
    G --> H2[Mathematical Validation Matrix]
    G --> H3[Automated Strategic Playbook]