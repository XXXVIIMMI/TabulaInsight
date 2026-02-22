"""
Professional Exploratory Data Analysis Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles any tabular dataset (CSV, Excel, JSON, Parquet, TSV).
Auto-detects column types and builds every section dynamically.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONSTANTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPORTED_TYPES = ["csv", "xlsx", "xls", "json", "txt", "tsv", "parquet"]
MAX_CAT_UNIQUE = 200
MAX_FILTER_COLS = 8
SAMPLE_THRESHOLD = 100_000
SAMPLE_SIZE = 50_000
COLOR_PALETTE = px.colors.qualitative.Set2
THEME = "plotly_white"

# ━━━━━━━━━━━━━━━━━━━━━━━━ PAGE CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="EDA Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        div.block-container { padding-top: 1rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] { font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 8px 20px;
        }
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Exploratory Data Analysis Dashboard")
st.caption("Upload any tabular dataset and get instant, comprehensive analysis.")

# ━━━━━━━━━━━━━━━━━━━━━━ DATA LOADING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.cache_data(show_spinner="Reading file …")
def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        if name.endswith(".json"):
            return pd.read_json(uploaded_file)
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()


file = st.file_uploader("Upload a dataset", type=SUPPORTED_TYPES,
                         help="CSV, Excel, JSON, Parquet, TSV, TXT")
if file:
    st.success(f"Loaded **{file.name}**")
    raw_df = load_file(file)
else:
    st.markdown(
        """
        <div style="text-align:center; padding:4rem 1rem;">
            <h2 style="color:#667eea;">Upload a dataset to get started</h2>
            <p style="font-size:1.1rem; color:#888;">
                Drop a <b>CSV</b>, <b>Excel</b>, <b>JSON</b>, <b>Parquet</b>, or <b>TSV</b> file above<br>
                and the dashboard will instantly generate a full analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━ COLUMN CLASSIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━


def classify_columns(df: pd.DataFrame):
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    boo = df.select_dtypes(include="bool").columns.tolist()
    for c in list(cat):
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.80:
                df[c] = parsed
                dt.append(c)
                cat.remove(c)
        except Exception:
            pass
    return num, cat, dt, boo


numeric_cols, categorical_cols, datetime_cols, bool_cols = classify_columns(raw_df)
all_cols = raw_df.columns.tolist()

# ━━━━━━━━━━━━━━━━━━━━ SIDEBAR FILTERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.header("🔍  Filters")

# Date range
date_col: str | None = None
if datetime_cols:
    date_col = st.sidebar.selectbox("Date column", datetime_cols, key="_f_dc")
    dmin = raw_df[date_col].dropna().min()
    dmax = raw_df[date_col].dropna().max()
    dr = st.sidebar.date_input("Date range", value=(dmin, dmax),
                                min_value=dmin, max_value=dmax, key="_f_dr")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        raw_df = raw_df[
            (raw_df[date_col] >= pd.to_datetime(dr[0]))
            & (raw_df[date_col] <= pd.to_datetime(dr[1]))
        ].copy()

# Numeric range sliders
st.sidebar.markdown("**Numeric ranges**")
for nc in numeric_cols[:3]:
    cmin, cmax = float(raw_df[nc].min()), float(raw_df[nc].max())
    if cmin < cmax:
        lo, hi = st.sidebar.slider(nc, cmin, cmax, (cmin, cmax), key=f"_fn_{nc}")
        raw_df = raw_df[(raw_df[nc] >= lo) & (raw_df[nc] <= hi)]

# Categorical multi-selects
st.sidebar.markdown("**Categories**")
cat_sel: dict[str, list] = {}
for cc in categorical_cols[:MAX_FILTER_COLS]:
    uniq = raw_df[cc].dropna().unique().tolist()
    if 1 < len(uniq) <= MAX_CAT_UNIQUE:
        sel = st.sidebar.multiselect(cc, sorted(map(str, uniq)), key=f"_fc_{cc}")
        if sel:
            cat_sel[cc] = sel

filtered_df = raw_df.copy()
for cc, vals in cat_sel.items():
    filtered_df = filtered_df[filtered_df[cc].astype(str).isin(vals)]

# Sampling guard
is_sampled = len(filtered_df) > SAMPLE_THRESHOLD
plot_df = filtered_df.sample(SAMPLE_SIZE, random_state=42) if is_sampled else filtered_df

st.sidebar.markdown("---")
st.sidebar.metric("Filtered rows", f"{len(filtered_df):,}")
st.sidebar.metric("Total rows", f"{len(raw_df):,}")
if is_sampled:
    st.sidebar.warning(f"Charts sampled to {SAMPLE_SIZE:,} rows for performance.")


# ━━━━━━━━━━━━━━━━━━━━━━━━ HELPER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def safe_plotly(fig, **kwargs):
    fig.update_layout(
        template=THEME,
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=40, b=30, l=30, r=20),
        colorway=COLOR_PALETTE,
    )
    st.plotly_chart(fig, use_container_width=True, **kwargs)


# ━━━━━━━━━━━━━━━━━━━ 1. DATA QUALITY PROFILE ━━━━━━━━━━━━━━━━━━━━━
st.header("1 · Data Quality Profile")

total_cells = filtered_df.shape[0] * filtered_df.shape[1]
missing_cells = int(filtered_df.isnull().sum().sum())
dup_rows = int(filtered_df.duplicated().sum())
mem_mb = filtered_df.memory_usage(deep=True).sum() / 1e6
quality_pct = round((1 - missing_cells / max(total_cells, 1)) * 100, 1)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rows", f"{filtered_df.shape[0]:,}")
m2.metric("Columns", f"{filtered_df.shape[1]:,}")
m3.metric("Missing cells", f"{missing_cells:,}",
          delta=f"{missing_cells / max(total_cells, 1) * 100:.1f}%", delta_color="inverse")
m4.metric("Duplicate rows", f"{dup_rows:,}",
          delta=f"{dup_rows / max(len(filtered_df), 1) * 100:.1f}%", delta_color="inverse")
m5.metric("Memory", f"{mem_mb:.1f} MB")
m6.metric("Completeness", f"{quality_pct}%")

tab_prev, tab_dtype, tab_stats, tab_miss, tab_dup, tab_card = st.tabs(
    ["Preview", "Data Types", "Statistics", "Missing Values", "Duplicates", "Cardinality"]
)

with tab_prev:
    st.dataframe(filtered_df.head(200), use_container_width=True, height=400)

with tab_dtype:
    type_df = pd.DataFrame({
        "Column": all_cols,
        "pandas dtype": filtered_df.dtypes.astype(str).values,
        "Detected": ["numeric" if c in numeric_cols
                      else "datetime" if c in datetime_cols
                      else "boolean" if c in bool_cols
                      else "categorical" for c in all_cols],
        "Non-null": filtered_df.notna().sum().values,
        "Null %": (filtered_df.isnull().mean() * 100).round(1).values,
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)

with tab_stats:
    desc = filtered_df.describe(include="all", percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc.index.name = "Column"
    st.dataframe(desc, use_container_width=True)

with tab_miss:
    miss = filtered_df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    miss["% Missing"] = (miss["Missing"] / max(len(filtered_df), 1) * 100).round(2)
    miss = miss.sort_values("% Missing", ascending=False)
    miss_nz = miss[miss["Missing"] > 0]
    if miss_nz.empty:
        st.success("No missing values! 🎉")
    else:
        st.dataframe(miss_nz, use_container_width=True, hide_index=True)
        fig = px.bar(miss_nz, x="Column", y="% Missing", text="% Missing",
                     color="% Missing", color_continuous_scale="OrRd",
                     title="Missing Value Distribution")
        safe_plotly(fig)

with tab_dup:
    if dup_rows == 0:
        st.success("No duplicate rows found! 🎉")
    else:
        st.warning(f"**{dup_rows:,}** duplicate rows ({dup_rows / max(len(filtered_df), 1) * 100:.1f}%)")
        with st.expander("Show duplicate rows"):
            st.dataframe(filtered_df[filtered_df.duplicated(keep=False)].head(500),
                         use_container_width=True)

with tab_card:
    card = pd.DataFrame({
        "Column": all_cols,
        "Unique": [filtered_df[c].nunique() for c in all_cols],
        "Total non-null": [filtered_df[c].notna().sum() for c in all_cols],
    })
    card["Uniqueness %"] = (card["Unique"] / card["Total non-null"].replace(0, 1) * 100).round(1)
    card["Type hint"] = card.apply(
        lambda r: "ID / key" if r["Uniqueness %"] > 95
        else "Low cardinality" if r["Unique"] <= 10
        else "Medium cardinality" if r["Unique"] <= 100
        else "High cardinality", axis=1)
    st.dataframe(card.sort_values("Unique", ascending=False),
                 use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━ 2. UNIVARIATE — NUMERIC ━━━━━━━━━━━━━━━━━━━━━
if numeric_cols:
    st.header("2 · Numeric Distributions")

    sk_df = pd.DataFrame({
        "Column": numeric_cols,
        "Mean": [filtered_df[c].mean() for c in numeric_cols],
        "Median": [filtered_df[c].median() for c in numeric_cols],
        "Std": [filtered_df[c].std() for c in numeric_cols],
        "Skewness": [filtered_df[c].skew() for c in numeric_cols],
        "Kurtosis": [filtered_df[c].kurtosis() for c in numeric_cols],
    }).round(3)
    with st.expander("Skewness & Kurtosis table"):
        st.dataframe(sk_df, use_container_width=True, hide_index=True)

    sel_num = st.multiselect("Columns to plot", numeric_cols,
                              default=numeric_cols[:min(4, len(numeric_cols))], key="_dist")
    dist_mode = st.radio("Plot type", ["Histogram + Box", "KDE", "ECDF"],
                          horizontal=True, key="_dm")

    if sel_num:
        n_cols = min(len(sel_num), 3)
        cols = st.columns(n_cols)
        for idx, cn in enumerate(sel_num):
            with cols[idx % n_cols]:
                if dist_mode == "Histogram + Box":
                    fig = px.histogram(plot_df, x=cn, marginal="box", title=cn, nbins=50)
                elif dist_mode == "KDE":
                    fig = px.histogram(plot_df, x=cn, marginal="rug",
                                       histnorm="probability density", title=cn, nbins=60)
                else:
                    fig = px.ecdf(plot_df, x=cn, title=f"ECDF — {cn}")
                safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━━ 3. UNIVARIATE — CATEGORICAL ━━━━━━━━━━━━━━━━━━
if categorical_cols:
    st.header("3 · Categorical Distributions")
    tn1, tn2, tn3 = st.columns(3)
    with tn1:
        tn_col = st.selectbox("Column", categorical_cols, key="_tn_col")
    with tn2:
        top_n = st.slider("Top N", 5, 50, 15, key="_tn_n")
    with tn3:
        tn_mode = st.radio("Chart", ["Bar", "Pie", "Both"], horizontal=True, key="_tn_mode")

    vc = filtered_df[tn_col].value_counts().head(top_n).reset_index()
    vc.columns = [tn_col, "Count"]
    vc["Pct"] = (vc["Count"] / vc["Count"].sum() * 100).round(1)

    if tn_mode in ("Bar", "Both"):
        c_left = st.columns(1)[0] if tn_mode == "Bar" else st.columns(2)[0]
        with c_left:
            fig = px.bar(vc, x=tn_col, y="Count", text="Pct", color=tn_col,
                         title=f"Top {top_n} — {tn_col}")
            fig.update_traces(texttemplate="%{text:.1f}%")
            safe_plotly(fig)
    if tn_mode in ("Pie", "Both"):
        c_right = st.columns(1)[0] if tn_mode == "Pie" else st.columns(2)[1]
        with c_right:
            fig = px.pie(vc, names=tn_col, values="Count", hole=0.4,
                         title=f"Top {top_n} — {tn_col} (share)")
            safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━ 4. CATEGORY × NUMERIC ━━━━━━━━━━━━━━━━━━━━━━━━
if categorical_cols and numeric_cols:
    st.header("4 · Category vs Numeric")
    cv1, cv2, cv3 = st.columns(3)
    with cv1:
        cat_col = st.selectbox("Category column", categorical_cols, key="_cv_cat")
    with cv2:
        num_col = st.selectbox("Numeric column", numeric_cols, key="_cv_num")
    with cv3:
        cv_agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], key="_cv_agg")

    grp = (filtered_df.groupby(cat_col, as_index=False)[num_col]
           .agg(cv_agg).sort_values(num_col, ascending=False))

    b_col, p_col = st.columns(2)
    with b_col:
        fig = px.bar(grp.head(20), x=cat_col, y=num_col, text_auto=",.2f",
                     color=num_col, color_continuous_scale="Tealgrn",
                     title=f"{cv_agg.title()} of {num_col} by {cat_col}")
        safe_plotly(fig)
    with p_col:
        fig = px.pie(grp.head(10), names=cat_col, values=num_col, hole=0.4,
                     title=f"Share of {num_col} by {cat_col}")
        safe_plotly(fig)

    with st.expander(f"View grouped data ({cat_col})"):
        st.dataframe(grp.style.background_gradient(cmap="Blues"), use_container_width=True)
        st.download_button("Download", grp.to_csv(index=False).encode("utf-8"),
                           "grouped.csv", "text/csv", key="_dl_grp")

# ━━━━━━━━━━━━━━━━━━ 5. BOX / VIOLIN / STRIP ━━━━━━━━━━━━━━━━━━━━━
if numeric_cols and categorical_cols:
    st.header("5 · Distribution Comparison")
    bv1, bv2, bv3 = st.columns(3)
    with bv1:
        bv_num = st.selectbox("Numeric", numeric_cols, key="_bv_num")
    with bv2:
        bv_cat = st.selectbox("Group by", categorical_cols, key="_bv_cat")
    with bv3:
        bv_mode = st.radio("Type", ["Box", "Violin", "Strip"], horizontal=True, key="_bv_mode")

    bv_data = plot_df.dropna(subset=[bv_num, bv_cat])
    top15 = bv_data[bv_cat].value_counts().head(15).index
    bv_data = bv_data[bv_data[bv_cat].isin(top15)]

    if bv_mode == "Box":
        fig = px.box(bv_data, x=bv_cat, y=bv_num, color=bv_cat, points="outliers",
                     title=f"{bv_num} by {bv_cat}")
    elif bv_mode == "Violin":
        fig = px.violin(bv_data, x=bv_cat, y=bv_num, color=bv_cat, box=True, points="all",
                        title=f"{bv_num} by {bv_cat}")
    else:
        fig = px.strip(bv_data, x=bv_cat, y=bv_num, color=bv_cat,
                       title=f"{bv_num} by {bv_cat}")
    fig.update_layout(showlegend=False)
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━━ 6. TIME SERIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if datetime_cols and numeric_cols:
    st.header("6 · Time Series Analysis")
    ts1, ts2, ts3 = st.columns(3)
    with ts1:
        ts_date = st.selectbox("Date column", datetime_cols, key="_ts_date")
    with ts2:
        ts_val = st.selectbox("Value column", numeric_cols, key="_ts_val")
    with ts3:
        ts_gran = st.selectbox("Granularity",
                                ["Day", "Week", "Month", "Quarter", "Year"],
                                index=2, key="_ts_gran")

    freq_map = {"Day": "D", "Week": "W", "Month": "ME", "Quarter": "QE", "Year": "YE"}
    ts_df = filtered_df.dropna(subset=[ts_date, ts_val]).copy()
    ts_df = ts_df.set_index(ts_date).resample(freq_map[ts_gran])[[ts_val]].sum().reset_index()

    ts_tab1, ts_tab2 = st.tabs(["Line Chart", "Area + Cumulative"])
    with ts_tab1:
        fig = px.line(ts_df, x=ts_date, y=ts_val,
                      title=f"{ts_val} over time ({ts_gran})", markers=True)
        safe_plotly(fig)
    with ts_tab2:
        ts_df["Cumulative"] = ts_df[ts_val].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df[ts_date], y=ts_df[ts_val],
                                 mode="lines", fill="tozeroy", name=ts_val))
        fig.add_trace(go.Scatter(x=ts_df[ts_date], y=ts_df["Cumulative"],
                                 mode="lines", name="Cumulative", yaxis="y2"))
        fig.update_layout(
            yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
            title=f"{ts_val} — Volume & Cumulative ({ts_gran})",
        )
        safe_plotly(fig)

    with st.expander("View time-series data"):
        st.dataframe(ts_df, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━ 7. CORRELATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(numeric_cols) >= 2:
    st.header("7 · Correlation Analysis")
    corr_method = st.radio("Method", ["pearson", "spearman", "kendall"],
                            horizontal=True, key="_corr_m")
    corr = filtered_df[numeric_cols].corr(method=corr_method)

    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    aspect="auto", title=f"{corr_method.title()} Correlation Matrix")
    fig.update_layout(height=max(400, 40 * len(numeric_cols)))
    safe_plotly(fig)

    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack().reset_index())
    pairs.columns = ["Var A", "Var B", "Correlation"]
    pairs["Abs"] = pairs["Correlation"].abs()
    pairs = pairs.sort_values("Abs", ascending=False).head(15).drop(columns="Abs")
    with st.expander("Top correlated pairs"):
        st.dataframe(pairs, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━ 8. SCATTER PLOT EXPLORER ━━━━━━━━━━━━━━━━━━━━
if len(numeric_cols) >= 2:
    st.header("8 · Scatter Plot Explorer")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        x_c = st.selectbox("X axis", numeric_cols, index=0, key="_sc_x")
    with sc2:
        y_c = st.selectbox("Y axis", numeric_cols,
                            index=min(1, len(numeric_cols) - 1), key="_sc_y")
    with sc3:
        clr = st.selectbox("Color (optional)", [None] + categorical_cols, key="_sc_clr")
    with sc4:
        sz = st.selectbox("Size (optional)", [None] + numeric_cols, key="_sc_sz")

    kw: dict[str, Any] = dict(x=x_c, y=y_c, opacity=0.55,
                               title=f"{y_c} vs {x_c}",
                               hover_data=plot_df.columns.tolist()[:5])
    if clr:
        kw["color"] = clr
    if sz:
        kw["size"] = sz
        kw["size_max"] = 18
    if len(plot_df) < 50_000:
        kw["trendline"] = "ols"

    fig = px.scatter(plot_df, **kw)
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━━ 9. OUTLIER DETECTION ━━━━━━━━━━━━━━━━━━━━━━━━
if numeric_cols:
    st.header("9 · Outlier Detection (IQR)")
    oc1, oc2 = st.columns([2, 1])
    with oc1:
        out_col = st.selectbox("Column", numeric_cols, key="_out_col")
    with oc2:
        iqr_mult = st.number_input("IQR multiplier", 1.0, 5.0, 1.5, 0.1, key="_iqr_m")

    q1 = filtered_df[out_col].quantile(0.25)
    q3 = filtered_df[out_col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    outliers = filtered_df[(filtered_df[out_col] < lo) | (filtered_df[out_col] > hi)]

    om1, om2, om3, om4, om5 = st.columns(5)
    om1.metric("Q1", f"{q1:,.2f}")
    om2.metric("Q3", f"{q3:,.2f}")
    om3.metric("IQR", f"{iqr:,.2f}")
    om4.metric("Bounds", f"[{lo:,.1f}, {hi:,.1f}]")
    om5.metric("Outliers",
               f"{len(outliers):,}  ({len(outliers) / max(len(filtered_df), 1) * 100:.1f}%)")

    fig = px.box(filtered_df, y=out_col, points="outliers",
                 title=f"Outliers in {out_col}  (×{iqr_mult} IQR)")
    safe_plotly(fig)
    with st.expander(f"View {len(outliers):,} outlier rows"):
        st.dataframe(outliers.head(1000), use_container_width=True)

# ━━━━━━━━━━━━━━━━━ 10. PAIR PLOT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(numeric_cols) >= 2:
    st.header("10 · Pair Plot")
    pp_cols = st.multiselect("Columns (2–5 recommended)", numeric_cols,
                              default=numeric_cols[:min(3, len(numeric_cols))], key="_pp")
    pp_color = st.selectbox("Color by", [None] + categorical_cols, key="_pp_clr")
    if len(pp_cols) >= 2:
        sample = plot_df[pp_cols + ([pp_color] if pp_color else [])].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        fig = px.scatter_matrix(sample, dimensions=pp_cols,
                                color=pp_color if pp_color else None,
                                opacity=0.45, title="Scatter Matrix")
        fig.update_layout(height=700)
        fig.update_traces(diagonal_visible=True, marker_size=3)
        safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━ 11. TREEMAP & SUNBURST ━━━━━━━━━━━━━━━━━━━━━━
if len(categorical_cols) >= 2 and numeric_cols:
    st.header("11 · Hierarchical View")
    tr1, tr2, tr3 = st.columns(3)
    with tr1:
        h_cols = st.multiselect("Hierarchy levels", categorical_cols,
                                 default=categorical_cols[:min(2, len(categorical_cols))],
                                 key="_tr_lvl")
    with tr2:
        h_val = st.selectbox("Value", numeric_cols, key="_tr_val")
    with tr3:
        h_mode = st.radio("Chart", ["Treemap", "Sunburst"], horizontal=True, key="_tr_mode")

    if h_cols:
        td = filtered_df.dropna(subset=h_cols + [h_val])
        if h_mode == "Treemap":
            fig = px.treemap(td, path=h_cols, values=h_val, color=h_val,
                             color_continuous_scale="Viridis",
                             title=f"Treemap — {h_val} by {' → '.join(h_cols)}")
        else:
            fig = px.sunburst(td, path=h_cols, values=h_val, color=h_val,
                              color_continuous_scale="Viridis",
                              title=f"Sunburst — {h_val} by {' → '.join(h_cols)}")
        fig.update_layout(height=650)
        safe_plotly(fig)

# ━━━━━━━━━━━━━━━ 12. GROUPED AGGREGATION ━━━━━━━━━━━━━━━━━━━━━━━━
if categorical_cols and numeric_cols:
    st.header("12 · Grouped Aggregation")
    ga1, ga2, ga3 = st.columns(3)
    with ga1:
        ga_grp = st.multiselect("Group by", categorical_cols,
                                  default=[categorical_cols[0]], key="_ga_grp")
    with ga2:
        ga_vals = st.multiselect("Values", numeric_cols,
                                   default=numeric_cols[:min(2, len(numeric_cols))],
                                   key="_ga_vals")
    with ga3:
        ga_agg = st.selectbox("Aggregation",
                               ["sum", "mean", "median", "count", "min", "max", "std"],
                               key="_ga_agg")

    if ga_grp and ga_vals:
        agg_df = filtered_df.groupby(ga_grp, as_index=False)[ga_vals].agg(ga_agg)
        st.dataframe(agg_df.style.background_gradient(cmap="YlGnBu"),
                     use_container_width=True)

        if len(ga_grp) == 1:
            fig = px.bar(agg_df.sort_values(ga_vals[0], ascending=False).head(25),
                         x=ga_grp[0], y=ga_vals[0], text_auto=",.2f",
                         color=ga_vals[0], color_continuous_scale="Tealgrn",
                         title=f"{ga_agg.title()} of {ga_vals[0]} by {ga_grp[0]}")
            safe_plotly(fig)
        elif len(ga_grp) >= 2:
            fig = px.bar(agg_df.sort_values(ga_vals[0], ascending=False).head(40),
                         x=ga_grp[0], y=ga_vals[0], color=ga_grp[1],
                         barmode="group", text_auto=",.0f",
                         title=f"{ga_agg.title()} of {ga_vals[0]} by "
                               f"{ga_grp[0]} & {ga_grp[1]}")
            safe_plotly(fig)

        st.download_button("Download aggregation",
                           agg_df.to_csv(index=False).encode("utf-8"),
                           "aggregation.csv", "text/csv", key="_dl_agg")

# ━━━━━━━━━━━━━━━━━━ 13. CROSS-TABULATION ━━━━━━━━━━━━━━━━━━━━━━━━
if len(categorical_cols) >= 2:
    st.header("13 · Cross-Tabulation")
    xt1, xt2, xt3 = st.columns(3)
    with xt1:
        xt_row = st.selectbox("Row", categorical_cols, index=0, key="_xt_r")
    with xt2:
        xt_col_sel = st.selectbox(
            "Column",
            [c for c in categorical_cols if c != xt_row] or categorical_cols,
            key="_xt_c",
        )
    with xt3:
        xt_norm = st.selectbox("Normalize", [None, "index", "columns", "all"],
                                key="_xt_n")

    ct_data = filtered_df.copy()
    for c in [xt_row, xt_col_sel]:
        top = ct_data[c].value_counts().head(15).index
        ct_data = ct_data[ct_data[c].isin(top)]

    ct = pd.crosstab(ct_data[xt_row], ct_data[xt_col_sel],
                     normalize=xt_norm if xt_norm else False)
    if xt_norm:
        ct = (ct * 100).round(1)
    st.dataframe(ct.style.background_gradient(cmap="PuBu"), use_container_width=True)

    fig = px.imshow(ct, text_auto=".1f" if xt_norm else ".0f",
                    color_continuous_scale="Blues",
                    title=f"Cross-tab: {xt_row} × {xt_col_sel}")
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━ 14. RADAR CHART ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(numeric_cols) >= 3 and categorical_cols:
    st.header("14 · Radar / Spider Chart")
    rd1, rd2 = st.columns(2)
    with rd1:
        rd_cat = st.selectbox("Compare by", categorical_cols, key="_rd_cat")
    with rd2:
        rd_nums = st.multiselect("Dimensions (3–8)", numeric_cols,
                                   default=numeric_cols[:min(5, len(numeric_cols))],
                                   key="_rd_nums")

    if len(rd_nums) >= 3:
        rd_top = filtered_df[rd_cat].value_counts().head(6).index
        rd_agg = (filtered_df[filtered_df[rd_cat].isin(rd_top)]
                  .groupby(rd_cat)[rd_nums].mean())
        rd_norm = (rd_agg - rd_agg.min()) / (rd_agg.max() - rd_agg.min() + 1e-9)

        fig = go.Figure()
        for cat_val in rd_norm.index:
            vals = rd_norm.loc[cat_val].tolist()
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=rd_nums + [rd_nums[0]],
                fill="toself", name=str(cat_val), opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Radar — mean of {len(rd_nums)} metrics by {rd_cat}",
        )
        safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━ 15. PARALLEL COORDINATES ━━━━━━━━━━━━━━━━━━━━━
if len(numeric_cols) >= 3:
    st.header("15 · Parallel Coordinates")
    pc_cols = st.multiselect("Axes", numeric_cols,
                              default=numeric_cols[:min(5, len(numeric_cols))],
                              key="_pc")
    pc_color = st.selectbox("Color axis", numeric_cols, key="_pc_clr")
    if len(pc_cols) >= 3:
        sample_pc = plot_df[pc_cols].dropna()
        fig = px.parallel_coordinates(sample_pc, dimensions=pc_cols, color=pc_color,
                                       color_continuous_scale="Viridis",
                                       title="Parallel Coordinates Plot")
        safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━━ 16. STACKED AREA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if datetime_cols and numeric_cols and categorical_cols:
    st.header("16 · Stacked Area Chart")
    sa1, sa2, sa3 = st.columns(3)
    with sa1:
        sa_date = st.selectbox("Date", datetime_cols, key="_sa_date")
    with sa2:
        sa_val = st.selectbox("Value", numeric_cols, key="_sa_val")
    with sa3:
        sa_cat = st.selectbox("Stack by", categorical_cols, key="_sa_cat")

    sa_df = filtered_df.dropna(subset=[sa_date, sa_val, sa_cat]).copy()
    sa_top = sa_df[sa_cat].value_counts().head(8).index
    sa_df = sa_df[sa_df[sa_cat].isin(sa_top)]
    sa_df["_period"] = sa_df[sa_date].dt.to_period("M").dt.to_timestamp()
    sa_grp = sa_df.groupby(["_period", sa_cat], as_index=False)[sa_val].sum()

    fig = px.area(sa_grp, x="_period", y=sa_val, color=sa_cat,
                  title=f"{sa_val} over time by {sa_cat}",
                  labels={"_period": "Period"})
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━ 17. PIVOT HEATMAP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(categorical_cols) >= 2 and numeric_cols:
    st.header("17 · Pivot Heatmap")
    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        hm_row = st.selectbox("Row", categorical_cols, index=0, key="_hm_r")
    with hm2:
        hm_col = st.selectbox(
            "Column",
            [c for c in categorical_cols if c != hm_row] or categorical_cols,
            key="_hm_c",
        )
    with hm3:
        hm_val = st.selectbox("Value", numeric_cols, key="_hm_v")

    hm_data = filtered_df.copy()
    for c in [hm_row, hm_col]:
        tops = hm_data[c].value_counts().head(20).index
        hm_data = hm_data[hm_data[c].isin(tops)]

    pivot = hm_data.pivot_table(index=hm_row, columns=hm_col,
                                 values=hm_val, aggfunc="sum")
    fig = px.imshow(pivot, text_auto=",.0f", color_continuous_scale="YlOrRd",
                    aspect="auto",
                    title=f"Sum of {hm_val}: {hm_row} × {hm_col}")
    fig.update_layout(height=max(400, 30 * len(pivot)))
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━ 18. FUNNEL CHART ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if categorical_cols and numeric_cols:
    st.header("18 · Funnel Chart")
    fn1, fn2 = st.columns(2)
    with fn1:
        fn_cat = st.selectbox("Stage column", categorical_cols, key="_fn_cat")
    with fn2:
        fn_val = st.selectbox("Value column", numeric_cols, key="_fn_val")

    fn_data = (filtered_df.groupby(fn_cat, as_index=False)[fn_val]
               .sum().sort_values(fn_val, ascending=False).head(10))
    fig = px.funnel(fn_data, x=fn_val, y=fn_cat,
                    title=f"Funnel — {fn_val} by {fn_cat}")
    safe_plotly(fig)

# ━━━━━━━━━━━━━━━━━━━━━ 19. EXPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.header("19 · Export Data")
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("⬇ Download CSV",
                       filtered_df.to_csv(index=False).encode("utf-8"),
                       "filtered_data.csv", "text/csv", key="_dl_csv")
with dl2:
    st.download_button("⬇ Download JSON",
                       filtered_df.to_json(orient="records",
                                           date_format="iso").encode("utf-8"),
                       "filtered_data.json", "application/json", key="_dl_json")
with dl3:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Filtered")
    st.download_button(
        "⬇ Download Excel", buf.getvalue(), "filtered_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="_dl_xlsx",
    )

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Plotly · Professional EDA Dashboard")
