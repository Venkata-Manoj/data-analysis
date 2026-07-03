# 📊 Customer Segmentation Dashboard (RFM Clustering)

**Project 1** of the Data Analysis Portfolio — an interactive dashboard for exploring customer segments generated via **RFM (Recency, Frequency, Monetary)** clustering.

## 🧠 What it does

Customer segmentation using KMeans clustering on the **UCI Online Retail** dataset. The pipeline computes R, F, and M scores for each customer, clusters them with KMeans, then serves results via an interactive Streamlit dashboard.

## 📂 Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard — filters, KPIs, Plotly charts |
| `rfm_segments.csv` | Precomputed RFM dataset with cluster assignments |
| `segmentation.ipynb` | Notebook: feature engineering + KMeans training + CSV export |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated charts and artifacts *(gitignored)* |

## 🔬 Pipeline

```
Load Data → Compute RFM → Cap Outliers → Standardize → KMeans → Export CSV → Streamlit Dashboard
```

1. **Load** the UCI Online Retail dataset
2. **Compute** per-customer:
   - **Recency**: days since last purchase
   - **Frequency**: number of unique invoices
   - **Monetary**: total spend
3. **Cap** outliers (99th percentile)
4. **Standardize** features
5. **Train KMeans** (k=4)
6. **Export** results to `rfm_segments.csv`
7. **Serve** via Streamlit dashboard

## 🚀 How to Run

```bash
cd customer-rfm-segmentation
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`

## 📈 Dashboard Features

- **Sidebar filters**: Segment, Recency range, Monetary range
- **Visuals** (Plotly):
  - Segment distribution (pie chart)
  - RFM 3D scatter (Recency × Frequency × Monetary)
  - Average RFM per segment (grouped bar chart)
- **Data table**: preview filtered results

## 🏷️ Segment Labels

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | 🏆 Champions | Top customers — recent, frequent, high spend |
| 1 | ❌ Lost | Inactive customers — low recency |
| 2 | 🌱 New/Promising | Recent but low frequency/spend |
| 3 | ⚠️ At Risk — Big Spenders | High spend but dropping recency |

## 📝 License

MIT
