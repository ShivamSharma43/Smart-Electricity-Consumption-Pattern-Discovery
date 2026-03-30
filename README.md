# AI304 Unsupervised Learning Lab — Smart Electricity Consumption Pattern Discovery

## Overview

This repository contains five Jupyter notebooks implementing a complete unsupervised learning pipeline on the UCI Individual Household Electric Power Consumption dataset. The goal is to discover household electricity consumption archetypes and detect anomalous usage events.

---

## Dataset

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
**File**: `household_power_consumption.txt`  
**Size**: ~2 million records (minute-level, Dec 2006 – Nov 2010)  
**Format**: Semicolon-delimited (`;`), missing values encoded as `?`

**Download and place the file** in the same directory as the notebooks before running.

---

## Execution Order

Run the notebooks **in the following sequence**. Each notebook saves intermediate files consumed by the next.

| # | Notebook | Input | Output |
|---|----------|-------|--------|
| 1 | `notebook1_preprocessing.ipynb` | `household_power_consumption.txt` | `cleaned_power_data.csv` |
| 2 | `notebook2_feature_engineering.ipynb` | `cleaned_power_data.csv` | `daily_features.csv`, `scaled_data.npy` |
| 3 | `notebook3_modelling.ipynb` | `daily_features.csv`, `scaled_data.npy` | `clustered_features.csv` |
| 4 | `notebook4_visualisation.ipynb` | `clustered_features.csv`, `scaled_data.npy` | `final_labelled_features.csv` |
| 5 | `notebook5_interpretation.ipynb` | `final_labelled_features.csv` | Final analysis (no file output) |

---

## Notebook Descriptions

### Notebook 1 — Data Understanding & Preprocessing
- Loads raw dataset (2M+ rows, semicolon-delimited)
- Replaces `?` with `NaN`, drops missing rows (~1.25%)
- Converts columns to `float64`
- Creates `datetime` index for time-series operations
- Visualises raw power trends and feature distributions
- Outputs: `cleaned_power_data.csv`

### Notebook 2 — Feature Engineering
- Aggregates minute-level data to daily resolution using `resample('D')`
- Engineers 10 features: mean/max/min/std power, peak-to-average ratio, sub-metering averages, voltage statistics
- Adds day-of-week and weekend indicator
- Applies `StandardScaler` for normalisation
- Outputs: `daily_features.csv`, `scaled_data.npy`

### Notebook 3 — Unsupervised Modelling & Clustering
- Selects optimal k using **Elbow Method** and **Silhouette Score** (k=2 to 10)
- Applies **K-Means**, **K-Medoids (PAM)**, **DBSCAN**, **Agglomerative Clustering**
- Plots dendrogram for hierarchical method
- Uses k-distance graph to tune DBSCAN `eps`
- Compares models by silhouette score
- Outputs: `clustered_features.csv`

### Notebook 4 — Visualisation & Analysis
- Scatter plots for all 4 algorithms (mean vs max power, coloured by cluster)
- PCA 2D projections of clusters
- Time-series plots coloured by cluster label
- Monthly cluster trend lines
- Feature heatmap showing cluster profiles
- DBSCAN noise point analysis (noise plotted separately in black)
- Isolation Forest anomaly detection with time-series overlay
- Outputs: `final_labelled_features.csv`

### Notebook 5 — Interpretation, Societal Relevance & Ethics
- Cluster archetype profiling (radar chart + feature comparison table)
- Weekend vs weekday distribution across clusters
- Anomaly month-wise analysis
- Real-world interpretation of each cluster archetype
- Societal benefits table
- Ethical considerations (privacy, fairness, consent, security)

---

## Software Environment

### Python Version
```
Python 3.10.x (recommended)
```

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | 1.26.4 | Numerical operations |
| `pandas` | ≥1.5 | Data manipulation |
| `scikit-learn` | ≥1.2 | Clustering, scaling, anomaly detection |
| `scikit-learn-extra` | ≥0.3 | K-Medoids (PAM) |
| `scipy` | ≥1.10 | Dendrogram, hierarchical linkage |
| `matplotlib` | ≥3.6 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualisations |

### Installation

```bash
pip install numpy==1.26.4
pip install scikit-learn scikit-learn-extra scipy matplotlib seaborn pandas
```

Or in Google Colab (run once at the top of Notebook 1):
```python
!pip install numpy==1.26.4 scikit-learn-extra --quiet
```

---

## Running on Google Colab

1. Open each notebook in Colab.
2. Run the first cell in Notebook 1 to install dependencies.
3. Use the `files.upload()` cell to upload `household_power_consumption.txt`.
4. Run all cells in order.
5. Subsequent notebooks will load files saved by the previous notebook.  
   Use `files.download('filename.csv')` if you need to transfer between sessions.

---

## Running Locally (Jupyter)

1. Install dependencies (see above).
2. Place `household_power_consumption.txt` in the project directory.
3. Open a terminal and run:
   ```bash
   jupyter notebook
   ```
4. Open and run each notebook in order (1 → 5).

---

## Models Used

| Model | Type | Library |
|-------|------|---------|
| K-Means | Centroid-based | `sklearn.cluster.KMeans` |
| K-Medoids | Medoid-based | `sklearn_extra.cluster.KMedoids` |
| DBSCAN | Density-based | `sklearn.cluster.DBSCAN` |
| Agglomerative | Hierarchical | `sklearn.cluster.AgglomerativeClustering` |
| Isolation Forest | Anomaly detection | `sklearn.ensemble.IsolationForest` |

---
