# Smart Food Delivery — Streamlit Dashboard

## Quick Start

### Step 1 — Make sure Python 3.8+ is installed
Download from https://python.org if needed.

### Step 2 — Run the dashboard

**Windows:**
Double-click `run_windows.bat`

**Mac / Linux:**
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

**Manual (any OS):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Step 3 — Open browser
Go to **http://localhost:8501**

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| EDA & Descriptive | 8 charts — adoption by city, persona distribution, NPS, surge attitude, importance heatmap |
| Correlation | Pearson heatmap, scatter matrix, top predictors of adoption |
| Clustering | K-Means with elbow curve, PCA scatter, cluster profiles |
| Classification | Random Forest / GBM / Logistic Regression with ROC, confusion matrix, feature importance |
| Regression | Spend prediction with RF Regressor + Linear Regression, R², RMSE |
| Association Rules | Apriori with adjustable support/confidence/lift — cuisine, addons, triggers, festivals |
| Business Insights | Viability scorecard, revenue bubble chart, strategic recommendations |

---

## Files Required
- `app.py` — main dashboard
- `food_delivery_survey_raw_2000.csv` — dataset (must be in same folder)
- `requirements.txt` — Python dependencies
- `run_windows.bat` — Windows launcher
- `run_mac_linux.sh` — Mac/Linux launcher

---

## Sidebar Filters
All charts update dynamically when you filter by:
- City Tier (Metro / Tier-2 / Tier-3 / Rural)
- Persona (P1–P6)
- Income Bracket

---
*Smart Food Delivery Optimization Platform — India Market Validation*
*Project by Aditya*
