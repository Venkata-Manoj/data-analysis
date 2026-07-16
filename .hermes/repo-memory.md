# data-analysis Repository Memory

## Repo Purpose
Portfolio of independent data analysis and machine learning projects. Each project lives in its own subdirectory with a complete analysis narrative.

## CI
- `.github/workflows/ci.yml` validates: ruff lint, project directories, key files, Python syntax
- Validates all 6 projects: customer-rfm-segmentation, nlp-sentiment-analysis, house-price-prediction, wine-quality-classification, pm25-air-quality-forecasting, topic-modeling-newsgroups

## Key Conventions
- Each project in own subdirectory with README.md, requirements.txt, main analysis script
- Charts in charts/ subdirectory per project
- Data in data/ subdirectory (gitignored; scripts auto-download)
- Outputs in outputs/ subdirectory (gitignored)
- Master README.md has project index + chart gallery

## Projects
1. **customer-rfm-segmentation** — KMeans clustering + RFM scoring. Streamlit dashboard.
2. **nlp-sentiment-analysis** — TF-IDF + Logistic Regression (88% acc, 0.953 AUC) on IMDB reviews. Streamlit dashboard.
3. **house-price-prediction** — Gradient Boosting (R²=0.836) on California Housing. Feature engineering with IncomePerRoom. 8 charts.
4. **wine-quality-classification** [ADDED 2026-07-09] — Random Forest (AUC=0.955) on UCI Wine Quality. Binary + multi-class. 8 charts. Auto-download dataset.
5. **pm25-air-quality-forecasting** [ADDED 2026-07-10] — Multivariate time series forecasting. Linear Regression (R²=0.9465) beats XGBoost. 32 features, 8 charts, UCI Beijing PM2.5 dataset.
6. **topic-modeling-newsgroups** [ADDED 2026-07-16] — Unsupervised topic discovery on 20 Newsgroups (1,600 docs, 8 categories). Compares NMF, LDA, TruncatedSVD. LDA achieves best coherence (0.2017). 6 charts: word clouds, heatmaps, 3D topic space.

## Completed Work
- [2026-07-08] Initial setup: CI, project structure, README template
- [2026-07-09] Project 4: Wine Quality Classification
- [2026-07-10] Project 5: PM2.5 Air Quality Forecasting
- [2026-07-16] Project 6: Topic Modeling on 20 Newsgroups
  - 3 algorithms compared: NMF, LDA, TruncatedSVD (LSA)
  - Subsampled 200 docs/category for fast training (1,600 total)
  - LDA: 0.2017 coherence, NMF: 0.0374, LSA: 0.0451
  - 6 charts generated (word clouds, heatmaps, 3D space, histograms)
  - Master README updated with project entry + chart gallery
  - CI updated to validate new project
