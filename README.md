# Customer Analytics Desktop App

A cross-platform (Windows & Mac) desktop application for customer segmentation,
data exploration, and next best action analysis.

---

## Project Structure

```
customer-analytics/
├── main.py                        # Entry point
├── requirements.txt
├── assets/
│   └── style.qss                  # Global stylesheet
├── app/
│   ├── ui/
│   │   ├── main_window.py         # Shell: sidebar + page stack
│   │   └── pages/
│   │       ├── import_page.py     # File import & preview
│   │       ├── explore_page.py    # Column profiling
│   │       ├── segmentation_page.py  # Clustering & RFM segmentation
│   │       └── nba_page.py           # Next best action (stub)
│   ├── core/
│   │   ├── data_loader.py         # CSV / Excel / JSON / Parquet loading
│   │   ├── profiler.py            # Column statistics (pandas or DuckDB)
│   │   └── segmentation.py      # K-Means, GMM, K-Prototypes, RFM
│   └── utils/
│       └── app_state.py           # Shared dataset state across pages
└── data/
    └── samples/                   # Drop sample CSV files here for testing
```

---

## Features

### Import
- Supported formats: **CSV**, **Excel** (`.xlsx` / `.xls`), **JSON**, **Parquet**
- Drag-and-drop or file browser
- Background loading with progress indicator
- CSV and Parquet files over **500 MB** open in **DuckDB preview mode** (100-row preview in memory; full data queried via SQL for profiling). CSV encoding is taken from the import selector.

### Explore
- Per-column statistics: count, unique values, missing %, numeric summaries, top categories
- Auto-profiles when you open the page after importing data
- Uses pandas for in-memory datasets; DuckDB `SUMMARIZE` for large CSV / Parquet files

### Segmentation
- **K-Means** — numeric and categorical features (categorical columns are dummy-encoded)
- **Gaussian Mixture** — same feature handling as K-Means
- **K-Prototypes** — mixed numeric and categorical features without dummy encoding
- **RFM** — quartile tiering on Recency, Frequency, and Monetary columns
- All clustering methods write assignments to a shared **`cluster_label`** column; RFM uses **`rfm_segment`**
- Export segment assignments (`.xlsx`) and fitted model (`.pkl`)

> **Large CSV / Parquet (> 500 MB):** Explore runs via DuckDB SQL. **Clustering** fits on a configurable random sample (default 50k rows), then optionally assigns labels to the **full file** in batches. **RFM** uses full-dataset quartiles and streams tier assignment — no sampling.

### Next Best Action
- Placeholder page — planned for a future sprint

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python main.py
```

---

## Packaging as a standalone installer

### Windows (.exe)
```bash
pyinstaller --onefile --windowed --name "CustomerAnalytics" main.py
# Output: dist/CustomerAnalytics.exe
```

### Mac (.app)
```bash
pyinstaller --onefile --windowed --name "CustomerAnalytics" main.py
# Output: dist/CustomerAnalytics.app
```

> For a polished Windows installer, wrap the output `.exe` with **Inno Setup**.
> For Mac distribution, sign and notarize the `.app` bundle.

---

## Sprint Roadmap

| Sprint | Feature | Status |
|--------|---------|--------|
| 1 | Data Import (CSV, Excel, JSON, Parquet) | ✅ Done |
| 1 | Data Exploration & Profiling | ✅ Done |
| 2 | Customer Segmentation (K-Means, GMM, K-Prototypes, RFM) | ✅ Done |
| 2 | Export segment assignments & model | ✅ Done |
| 3 | Next Best Action Engine | 🔜 Planned |
| 4 | Export results to PDF | 🔜 Planned |
| 5 | Packaging & installer | 🔜 Planned |
