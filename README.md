# PCA-Based Compliance Risk Clustering — SIH26051

> **SIH Problem ID:** SIH26051 | Smart India Hackathon 2026  
> **Theme:** E-Commerce Product-Listing Compliance Analysis

---

## ⚠️ Disclaimer

This project provides **estimated risk signals** based on configurable,
rule-based scoring of available product-listing data fields.  
It **does not** detect or confirm legal violations, regulatory breaches, or
fraud. All flagged listings must be reviewed by a qualified compliance officer.  
The synthetic demonstration dataset is entirely artificial.

---

## Project Overview

E-commerce platforms handle millions of product listings. Ensuring that each
listing carries the required declarations (certifications, ingredient lists,
price disclosures, etc.) at scale is a major compliance challenge.

This application:
1. **Extracts** structured compliance features from product-listing data.
2. **Reduces** high-dimensional feature space using **Principal Component
   Analysis (PCA)**, preserving the most informative variance.
3. **Clusters** listings by compliance profile using **K-Means** (with optional
   DBSCAN).
4. **Scores** each listing with a transparent, configurable rule-based
   estimated-risk scoring method.
5. **Exports** enriched results including cluster labels, risk scores, and
   reason indicators.

---

## System Architecture

```
CSV / Generated Data
        │
        ▼
data_loader.py  ──►  data_validation.py
        │
        ▼
feature_engineering.py
  TF-IDF  ──►  TruncatedSVD ─┐
  OHE     ──►  dense         ├──► Dense Feature Matrix (X)
  Scaler  ──►  dense         ─┘
        │
        ▼
pca_analysis.py  ──►  X_reduced (n × k)
        │
        ├──► clustering.py   ──►  cluster labels
        │
        └──► risk_scoring.py ──►  risk_score, risk_label, risk_reasons
                │
                ▼
        export_utils.py  ──►  CSV / joblib outputs
                │
                ▼
        app.py (Streamlit Dashboard)
```

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique listing identifier |
| `category` | categorical | Product category |
| `title` | text | Product title (TF-IDF) |
| `description` | text | Product description (TF-IDF) |
| `declared_price` | float | Listed price |
| `declared_brand` | categorical | Brand name |
| `manufacturer` | string | Manufacturer name |
| `country_of_origin` | categorical | Country where manufactured |
| `certifications` | categorical | Comma-separated certification codes |
| `*_declared` | binary (0/1) | Whether a declaration field is filled |
| `missing_declaration_count` | int | Number of missing required declarations |
| `declaration_completeness_pct` | float | % of required declarations filled |
| `price_anomaly` | binary | 1 if price is unusually low/high |
| `is_duplicate_listing` | binary | 1 if listing appears duplicated |
| `title_desc_mismatch` | binary | 1 if title and description seem mismatched |
| `desc_length` | int | Character count of description |
| `listing_metadata_completeness` | float | % of optional metadata fields filled |
| `data_source` | string | `SYNTHETIC_DEMO` or `USER_UPLOAD` |

---

## PCA Explanation

**Principal Component Analysis (PCA)** finds the directions of maximum
variance in the feature matrix and projects data onto a smaller set of
orthogonal axes (principal components).

- The **scree plot** shows how much variance each component explains.
- The **cumulative variance curve** shows how many components are needed to
  reach a target (e.g. 90% of total variance).
- Component count is chosen automatically to meet the variance target, or
  can be set manually.

Because TF-IDF and one-hot encoding produce sparse, high-dimensional
representations, a **TruncatedSVD** step first reduces the sparse block
before the final dense PCA.

---

## Clustering Explanation

**K-Means** partitions the PCA-reduced listings into *k* groups by minimising
within-cluster variance. Cluster numbers have no inherent risk meaning.

**Silhouette score** (−1 to +1) measures how well each listing fits its own
cluster vs. neighbouring clusters; higher is better.

**Cluster labels** (e.g. "Cluster 0") are interpreted using the average risk
signals of their members, not by the cluster number itself.

---

## Risk Scoring Methodology

Risk is estimated with a transparent weighted-sum formula:

| Signal | Weight | Description |
|--------|--------|-------------|
| Missing required declaration fields | 0.30 | Normalised by max possible |
| Unusual declared price | 0.20 | Binary flag |
| Possible duplicate listing | 0.15 | Binary flag |
| Title-description mismatch | 0.15 | Binary flag |
| Declaration completeness < 50% | 0.10 | Binary threshold |
| Listing metadata completeness < 50% | 0.10 | Binary threshold |

The raw score is normalised to [0, 1] and mapped to three labels:

| Score range | Label |
|-------------|-------|
| [0, 0.30) | Lower estimated risk |
| [0.30, 0.60) | Moderate estimated risk |
| [0.60, 1.0] | Higher estimated risk |

All weights and thresholds are configurable via `RiskScoringConfig`.

---

## Project Structure

```
compliance-risk-clusters/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Synthetic data + CSV loader
│   ├── data_validation.py          # Schema & quality checks
│   ├── feature_engineering.py      # TF-IDF + OHE + Scaler pipeline
│   ├── pca_analysis.py             # PCA with variance reporting
│   ├── clustering.py               # K-Means + DBSCAN + silhouette
│   ├── risk_scoring.py             # Rule-based risk scorer
│   ├── visualization.py            # Plotly chart builders
│   └── export_utils.py             # CSV export + joblib persistence
├── tests/
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_pca_analysis.py
│   ├── test_clustering.py
│   └── test_risk_scoring.py
└── scripts/
    └── generate_sample_data.py     # CLI data generator
```

---

## Installation (Windows PowerShell)

### 1. Prerequisites
- Python 3.11 or 3.12 ([python.org](https://www.python.org/downloads/))
- Git for Windows ([git-scm.com](https://git-scm.com/downloads))

### 2. Clone the Repository
```powershell
git clone https://github.com/<your-username>/compliance-risk-clusters.git
cd compliance-risk-clusters
```

### 3. Create and Activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you see an execution-policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Application
```powershell
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## Running Tests

```powershell
python -m pytest -q
```

For verbose output with coverage:
```powershell
python -m pytest -v --tb=short --cov=src --cov-report=term-missing
```

---

## Generate Sample Data (CLI)

```powershell
python scripts/generate_sample_data.py --rows 500 --out data/sample_data.csv
```

---

## Demonstration Workflow

1. Open the app at `http://localhost:8501`.
2. Navigate to **📂 Dataset Management**.
3. Click **🚀 Generate & Load into App** (500 rows, seed 42).
4. Adjust cluster count (default 4) and click **🚀 Run Pipeline**.
5. Explore results in **📊 PCA Analysis** and **🎯 Risk Clustering**.
6. Filter listings in **🔎 Listing Explorer**.
7. Download the enriched CSV from **💾 Export & Model Management**.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit: command not found` | Activate the venv: `.\.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the venv |
| Script cannot run (policy error) | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Very slow on first run | TF-IDF and SVD fit can take a few seconds on 500+ rows |
| PCA warns about component count | Dataset may be small; the app auto-adjusts and warns |
| `FileNotFoundError: models/` | Run the pipeline at least once to create the model |

---

## Limitations

- Risk scoring is rule-based; it does not use a trained classifier.
- TF-IDF vocabulary is capped at 150 features per text column to control dimensionality.
- DBSCAN parameters (eps, min_samples) require manual tuning for different datasets.
- The synthetic dataset is for demonstration only and does not represent any real regulation or enforcement data.
