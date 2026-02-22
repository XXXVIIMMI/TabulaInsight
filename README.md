# TabulaInsight : Exploratory Data Analysis Dashboard

A professional, interactive **Exploratory Data Analysis** dashboard built with **Streamlit** and **Plotly**. Upload any tabular dataset and get instant, comprehensive visual analysis — no coding required.

---

## Features

The dashboard auto-detects column types (numeric, categorical, datetime, boolean) and dynamically builds **19 analysis sections**:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Data Quality Profile** | Row/column counts, missing values, duplicates, memory usage, completeness score, data types, cardinality analysis |
| 2 | **Numeric Distributions** | Histograms, box plots, KDE, ECDF with skewness & kurtosis stats |
| 3 | **Categorical Distributions** | Top-N bar charts and pie/donut charts for any categorical column |
| 4 | **Category vs Numeric** | Grouped aggregations (sum, mean, median, count) with bar & pie charts |
| 5 | **Distribution Comparison** | Box, violin, and strip plots grouped by category |
| 6 | **Time Series Analysis** | Line charts, area charts, and cumulative trends at configurable granularity (Day → Year) |
| 7 | **Correlation Analysis** | Heatmap with Pearson, Spearman, or Kendall methods and top correlated pairs |
| 8 | **Scatter Plot Explorer** | Interactive scatter with optional color, size, and OLS trendline |
| 9 | **Outlier Detection** | IQR-based outlier identification with adjustable multiplier |
| 10 | **Pair Plot** | Scatter matrix for multi-variable comparison |
| 11 | **Hierarchical View** | Treemap and sunburst charts for multi-level categorical breakdowns |
| 12 | **Grouped Aggregation** | Flexible group-by with multiple aggregation functions and downloadable results |
| 13 | **Cross-Tabulation** | Contingency tables with optional normalization and heatmap |
| 14 | **Radar / Spider Chart** | Normalized multi-metric comparison across categories |
| 15 | **Parallel Coordinates** | High-dimensional numeric data exploration |
| 16 | **Stacked Area Chart** | Time-based stacked area by category |
| 17 | **Pivot Heatmap** | Two-dimensional aggregated heatmap |
| 18 | **Funnel Chart** | Funnel visualization for staged/ranked data |
| 19 | **Export** | Download filtered data as CSV, JSON, or Excel |

### Additional Capabilities

- **Sidebar filters** — date range, numeric sliders, categorical multi-select
- **Smart sampling** — datasets over 100k rows are automatically sampled to 50k for chart performance
- **Multi-format upload** — CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet, TSV, TXT
- **Encoding auto-detection** — tries UTF-8, Latin-1, and CP1252
- **Bundled sample dataset** — ships with an Airbnb Open Data CSV for instant exploration

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io/) | Web app framework & UI |
| [Plotly](https://plotly.com/python/) | Interactive charting |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical operations |
| [statsmodels](https://www.statsmodels.org/) | OLS trendlines in scatter plots |
| [openpyxl](https://openpyxl.readthedocs.io/) / xlrd | Excel read/write support |

---

## Project Structure

```
EDA_Perform/
├── readme.md              # Project documentation
├── requirements.txt       # Python dependencies
└── src/
    ├── data/
    │   └── Airbnb_Open_Data.csv   # Bundled sample dataset
    └── project/
        └── dashboard.py           # Main Streamlit application
```

---

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
# Clone the repository
git clone <https://github.com/XXXVIIMMI/TabulaInsight.git>
cd TabulaInsight

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows


# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run src/project/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload a file** using the file uploader, or explore the bundled sample dataset loaded by default.
2. **Apply filters** in the sidebar — date ranges, numeric sliders, and categorical selectors.
3. **Browse analysis sections** — each section adapts to the columns present in your data.
4. **Export results** — download the filtered dataset as CSV, JSON, or Excel from section 19.

---
# Video 

https://github.com/user-attachments/assets/75d60a0f-cee2-48ab-9c53-8f8db9b305b0


## License

This project is open source and available under the [MIT License](LICENSE).

