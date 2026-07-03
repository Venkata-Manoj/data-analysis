# Customer Segmentation with RFM Clustering

Project 1 of the Data Analysis Portfolio. An interactive dashboard that groups customers into meaningful segments based on their buying behaviour. Built with Streamlit and Plotly.

## What This Does

Understand who your customers really are. This project takes transactional data and breaks customers into groups like "Champions" (your best, most loyal customers), "At Risk" (high spenders who haven't bought lately), and "Lost" (inactive for a while). Each segment gets its own profile with KPIs, charts, and actionable insights.

## How to Run

```bash
cd customer-rfm-segmentation
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
customer-rfm-segmentation/
├── README.md                <- This file
├── app.py                   <- Streamlit dashboard with filters, KPIs, and Plotly charts
├── requirements.txt         <- Python dependencies
└── data/                    <- Dataset files
    └── online_retail.xlsx   <- UCI Online Retail dataset
```

## Features

- RFM scoring (Recency, Frequency, Monetary) for each customer
- KMeans clustering to group customers into 4 segments
- Interactive filters by segment, purchase history, and spend range
- KPI cards with segment-level metrics
- Distribution charts and comparison views
- Download filtered results as CSV

## Customer Segments Found

| Segment | Label | Description |
|---------|-------|-------------|
| 0 | Champions | Top customers. Recent, frequent, high spend. |
| 1 | Lost | Inactive customers. Low recency scores. |
| 2 | Loyal Customers | Regular buyers who spend consistently. |
| 3 | At Risk - Big Spenders | High spend but dropping recency. Worth re-engaging. |

## License

MIT - feel free to use, modify, and share.
