import pandas as pd
import plotly.express as px
import streamlit as st

# Page config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Interactive view of customer RFM clusters")


# Load data (cached so it loads only once)
@st.cache_data
def load_data():
    df = pd.read_csv("rfm_segments.csv")
    return df


rfm = load_data()

# Map cluster numbers to business names (adjust based on your analysis)
cluster_names = {0: "Champions", 1: "Lost", 2: "New/Promising", 3: "At Risk – Big Spenders"}
rfm["Segment"] = rfm["Cluster"].map(cluster_names)

# Sidebar filters
st.sidebar.header("Filters")
selected_segments = st.sidebar.multiselect(
    "Select customer segments", options=rfm["Segment"].unique(), default=rfm["Segment"].unique()
)

recency_range = st.sidebar.slider(
    "Recency (days since last purchase)",
    int(rfm["Recency"].min()),
    int(rfm["Recency"].max()),
    (int(rfm["Recency"].min()), int(rfm["Recency"].max())),
)

monetary_range = st.sidebar.slider(
    "Monetary value (£)",
    float(rfm["Monetary"].min()),
    float(rfm["Monetary"].max()),
    (float(rfm["Monetary"].min()), float(rfm["Monetary"].max())),
)

# Filter data based on selections
filtered = rfm[
    (rfm["Segment"].isin(selected_segments))
    & (rfm["Recency"].between(recency_range[0], recency_range[1]))
    & (rfm["Monetary"].between(monetary_range[0], monetary_range[1]))
]

# KPI cards
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", filtered.shape[0])
col2.metric("Avg Recency (days)", round(filtered["Recency"].mean(), 1))
col3.metric("Avg Frequency", round(filtered["Frequency"].mean(), 1))
col4.metric("Avg Monetary (£)", round(filtered["Monetary"].mean(), 2))

# Visualizations
st.subheader("Segment Distribution")
fig_pie = px.pie(filtered, names="Segment", title="Customers per Segment")
st.plotly_chart(fig_pie, width="stretch")


st.subheader("RFM 3D Scatter Plot")
fig_3d = px.scatter_3d(
    filtered,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Segment",
    hover_data=["CustomerID"],
    title="Customer Segments in RFM Space",
)
st.plotly_chart(fig_3d, width="stretch")


st.subheader("Average RFM per Segment")
avg_rfm = filtered.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
fig_bar = px.bar(
    avg_rfm.melt(id_vars="Segment"),
    x="Segment",
    y="value",
    color="variable",
    barmode="group",
    title="Average Recency, Frequency, Monetary by Segment",
)
st.plotly_chart(fig_bar, width="stretch")


# Data table
st.subheader("Filtered Customer Data")
st.dataframe(filtered, height=300)
