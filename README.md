# Customer Segmentation Dashboard (RFM Clustering)

Interactive Streamlit dashboard for exploring customer segments generated using **RFM (Recency, Frequency, Monetary)** clustering.

## What this project contains
- **`app.py`**: Streamlit web app (filters + KPIs + Plotly charts)
- **`rfm_segments.csv`**: Precomputed RFM dataset with cluster assignments
- **`segmentation.ipynb`**: Notebook used to build the RFM features and train KMeans (and export the CSV)
- **`requirements.txt`**: Python dependencies

## How the segmentation works
1. Load the Online Retail dataset in the notebook.
2. Compute customer-level:
   - **Recency**: days since last purchase
   - **Frequency**: number of unique invoices
   - **Monetary**: total spend
3. Cap outliers and standardize features.
4. Train **KMeans** (default `k=4` in the notebook).
5. Export the resulting table to **`rfm_segments.csv`** with columns:
   - `CustomerID, Recency, Frequency, Monetary, Cluster`

## Running the dashboard
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start Streamlit
```bash
streamlit run app.py
```

Then open:
- `http://localhost:8501`

## Dashboard features
- Sidebar filters:
  - Segment (mapped from cluster id)
  - Recency range
  - Monetary range
- Visuals (Plotly):
  - Segment distribution (pie)
  - RFM 3D scatter (Recency vs Frequency vs Monetary)
  - Average RFM per segment (grouped bar)
- Data preview table for the filtered results

## Notes
- The app uses `st.cache_data` to avoid reloading `rfm_segments.csv` on each interaction.
- Cluster names are defined in `app.py`:
  - `0: Champions`
  - `1: Lost`
  - `2: New/Promising`
  - `3: At Risk – Big Spenders`

