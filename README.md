# Customer Analytics Desktop App

A cross-platform (Windows & Mac) desktop application for customer segmentation,
data exploration, and next best action analysis.

---

## Project Structure

```
customer_analytics/
├── main.py                        # Entry point
├── requirements.txt
├── assets/
│   └── style.qss                  # Global stylesheet
├── app/
│   ├── ui/
│   │   ├── main_window.py         # Shell: sidebar + page stack
│   │   └── pages/
│   │       ├── import_page.py     # ✅ Sprint 1 — File import & preview
│   │       ├── explore_page.py    # ✅ Sprint 1 — Column profiling
│   │       ├── segmentation_page.py  # 🔜 Sprint 2
│   │       └── nba_page.py           # 🔜 Sprint 3
│   ├── core/
│   │   ├── data_loader.py         # CSV / Excel / JSON loading
│   │   └── profiler.py            # Column statistics
│   └── utils/
│       └── app_state.py           # Shared DataFrame state
└── data/
    └── samples/                   # Drop sample CSV files here for testing
```

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
| 1 | Data Import (CSV, Excel, JSON) | ✅ Done |
| 1 | Data Exploration & Profiling   | ✅ Done |
| 2 | Customer Segmentation (K-Means / RFM) | 🔜 Next |
| 3 | Next Best Action Engine        | 🔜 Planned |
| 4 | Export results to Excel/PDF    | 🔜 Planned |
| 5 | Packaging & installer          | 🔜 Planned |
