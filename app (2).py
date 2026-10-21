"""
Hispanic Heritage Historical Data Explorer
A self-contained Streamlit classroom app for World Bank historical data and function analysis.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import io
import math
import re
import textwrap
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
import xlsxwriter

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current age-specific mortality rates remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, in current US dollars; not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants during the period; it is not an emigration count. Definitions and availability may vary."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people, based on the World Bank indicator definition and its source series."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP; coverage and reporting can vary by country."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, expressed in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.")
}
CONTEXTS = {
    "Business": ("Revenue", "currency units", "time", "years", "Quadratic growth/decline can be a local approximation; real revenue is not usually polynomial forever."),
    "Engineering": ("Position", "meters", "time", "seconds", "First derivative is velocity (m/s); second derivative is acceleration (m/s²)."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "A concentration curve is illustrative only and is not medical guidance."),
    "Sports science": ("Distance", "meters", "time", "seconds", "First derivative is speed along a one-dimensional path; units depend on the chosen variables."),
    "Biology": ("Population", "organisms", "time", "days", "Population models are simplified and can become unrealistic outside the observed interval."),
    "Bioengineering": ("Signal", "arbitrary units", "time", "seconds", "Signal derivatives describe how quickly the signal and its rate change."),
    "Economic forecasting": ("Indicator", "indicator units", "time", "years", "Forecasts are uncertain; polynomial extrapolations can behave implausibly."),
    "Weather": ("Temperature", "°C", "time", "hours", "A fitted curve is a teaching model, not an operational weather forecast."),
    "Custom": ("Output", "units", "Input", "units", "Describe the meaning and limitations of your custom model.")
}
SUGGESTED = {
    "Business": "a*x**2 + b*x + c", "Engineering": "a*x**2 + b*x + c",
    "Medical": "a*exp(-k*x) + c", "Sports science": "a*sin(b*x) + c",
    "Biology": "a*exp(k*x) + c", "Bioengineering": "a*sin(b*x) + c",
    "Economic forecasting": "a*x**2 + b*x + c", "Weather": "a*sin(b*x) + c",
    "Custom": "a*x + b"
}
X = sp.Symbol("x", real=True)
ALLOWED_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs
}
WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

def wb_fetch(country: str, indicator: str, start: int, end: int) -> pd.DataFrame:
    """Retrieve official World Bank API observations; missing values remain missing."""
    url = WORLD_BANK_BASE.format(country=country, indicator=indicator)
    params = {"date": f"{start}:{end}", "format": "json", "per_page": 20000}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank API returned an unexpected or empty response.")
    rows = []
    for item in payload[1]:
        if not isinstance(item, dict) or not item.get("date"):
            continue
        try:
            year = int(item["date"])
        except (TypeError, ValueError):
            continue
        val = item.get("value")
        rows.append({"year": year, "value": np.nan if val is None else float(val)})
    df = pd.DataFrame(rows, columns=["year", "value"])
    if df.empty:
        raise ValueError("No observations were returned for this country/indicator/year range.")
    return df.sort_values("year").drop_duplicates("year", keep="last").reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country: str, indicator: str, start: int, end: int) -> pd.DataFrame:
    return wb_fetch(country, indicator, start, end)

def parse_expr(expression: str):
    """Parse a constrained SymPy expression; only x, numeric constants and approved functions."""
    expression = expression.strip().replace("^", "**")
    if not expression or len(expression) > 180:
        raise ValueError("Enter a function (maximum 180 characters).")
    # Reject attributes, indexing, imports, and unsupported characters/names.
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s*]+", expression):
        raise ValueError("Use ordinary math notation, numbers, x, operators, and approved functions.")
    local = {"x": X, "pi": sp.pi, "E": sp.E}
    local.update(ALLOWED_FUNCTIONS)
    try:
        expr = sp.sympify(expression, locals=local, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse function: {exc}") from exc
    permitted = {X, sp.pi, sp.E, sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c"), sp.Symbol("k")}
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol and node not in permitted:
            raise ValueError(f"Unsupported symbol '{node}'. Use x and numeric parameters.")
        if node.is_Function and node.func not in set(ALLOWED_FUNCTIONS.values()):
            raise ValueError(f"Unsupported function '{node.func}'.")
    return expr

def safe_eval(expr, values):
    fn = sp.lambdify(X, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        result = np.asarray(fn(values), dtype=float)
    if result.ndim == 0:
        result = np.full(np.asarray(values).shape, float(result))
    return result

def validate_data(df: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    work = df.copy()
    if not {"year", "value"}.issubset(work.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must have year and value columns."]
    work = work[["year", "value"]].copy()
    year_num = pd.to_numeric(work["year"], errors="coerce")
    value_num = pd.to_numeric(work["value"], errors="coerce")
    if year_num.isna().any() or not np.all(np.isclose(year_num.dropna(), np.round(year_num.dropna()))):
        errors.append("Years must be whole-number integers.")
    work["year"] = year_num
    work["value"] = value_num
    work = work.dropna(subset=["year", "value"])
    if work["year"].duplicated().any():
        errors.append("Duplicate years found. Keep one row per year.")
    if not np.isfinite(work["value"]).all():
        errors.append("Values must be finite numbers.")
    work = work[np.isfinite(work["value"])]
    if indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate") and (work["value"] < 0).any():
        errors.append("This indicator cannot contain negative values.")
    if indicator_name == "Life expectancy at birth" and (work["value"] > 125).any():
        errors.append("Life expectancy values above 125 years are implausible; check the edit.")
    if indicator_name == "Unemployment, national estimate" and (work["value"] > 100).any():
        errors.append("Unemployment percentage cannot exceed 100.")
    work["year"] = work["year"].astype(int)
    return work.sort_values("year").reset_index(drop=True), errors

def metrics(y, pred):
    residual = np.asarray(y) - np.asarray(pred)
    return {
        "r2": float(1 - np.sum(residual**2) / np.sum((np.asarray(y)-np.mean(y))**2)) if np.sum((np.asarray(y)-np.mean(y))**2) else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "residuals": residual
    }

def fit_poly(df, degree):
    clean = df[["year", "value"]].dropna().sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree+1} distinct valid years; only {clean['year'].nunique()} available.")
    base_year = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base_year
    coeff = np.polynomial.polynomial.polyfit(t, clean["value"].to_numpy(dtype=float), degree)
    pred = np.polynomial.polynomial.polyval(t, coeff)
    return base_year, coeff, pred, metrics(clean["value"].to_numpy(), pred)

def fmt_num(value, precision=6):
    if not np.isfinite(value):
        return "undefined"
    return f"{value:.{precision}g}"

def poly_equation(coeff, base_year):
    terms = []
    for power, c in enumerate(coeff):
        terms.append(f"({c:.10g})" if power == 0 else f"({c:.10g})·t^{power}")
    return "y = " + " + ".join(terms) + f", where t = year − {base_year}"

def fit_holdout(df, degree):
    clean = df[["year", "value"]].dropna().sort_values("year")
    n = len(clean)
    test_n = max(1, int(round(n * 0.2)))
    if n - test_n < degree + 1 or n < degree + 3:
        return None
    train, test = clean.iloc[:-test_n], clean.iloc[-test_n:]
    base = int(train["year"].min())
    coeff = np.polynomial.polynomial.polyfit(train["year"].to_numpy()-base, train["value"].to_numpy(), degree)
    pred = np.polynomial.polynomial.polyval(test["year"].to_numpy()-base, coeff)
    return {"years": test["year"].to_numpy(), "actual": test["value"].to_numpy(), "predicted": pred,
            "mae": float(np.mean(np.abs(test["value"].to_numpy()-pred))), "n": len(test)}

def worldbank_link(country, indicator):
    return f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"

def build_pdf(title, metadata, equation, metric_lines, chart_bytes, preview, results, findings):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in metadata:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Model", styles["Heading2"]))
    story.append(Paragraph(equation.replace("&", "&amp;").replace("<", "&lt;"), styles["BodyText"]))
    for line in metric_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 8))
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=650, height=330))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for line in findings:
        story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;"), styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    data = [["Year", "Value"]] + [[str(int(r.year)), fmt_num(r.value, 8)] for r in preview.head(15).itertuples()]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), colors.lightgrey), ("GRID",(0,0),(-1,-1),0.3,colors.grey), ("FONTSIZE",(0,0),(-1,-1),8)]))
    story.append(tbl)
    if len(results):
        story.append(Spacer(1, 8))
        story.append(Paragraph("Calculated years", styles["Heading2"]))
        data = [["Year", "Estimate", "Range classification"]] + [[str(int(r.year)), fmt_num(r.estimate, 8), r.classification] for r in results.itertuples()]
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
        story.append(tbl)
    doc.build(story)
    return buf.getvalue()

# ---------------- UI ----------------
st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · Historical evidence + mathematical modeling")
st.info("Brazil is included as a Latin American comparison country. Country selection does not imply that Brazil is a Spanish-speaking country.")

now_year = date.today().year
start_year, end_year = now_year - 70, now_year - 1
with st.sidebar:
    st.header("Data and model controls")
    country_name = st.selectbox("Country", list(COUNTRIES), index=0)
    country_code = COUNTRIES[country_name]
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    indicator_code, units, definition = INDICATORS[indicator_name]
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_names = st.multiselect("Comparison countries", [c for c in COUNTRIES if c != country_name], default=[])
    printer_mode = st.checkbox("Printer-friendly view", value=False)

st.markdown(f"**Research question:** How has {indicator_name.lower()} changed in {country_name} over the historical period, and how well does a polynomial describe the observed pattern?")
st.markdown(f"**Requested period:** {start_year}–{end_year} (70 calendar years).")
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Indicator code:** `{indicator_code}`  \n**Source:** [World Bank API]({worldbank_link(country_code, indicator_code)})  \n**Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M')} (local app time)")

try:
    with st.spinner("Retrieving official World Bank observations…"):
        original = cached_wb_fetch(country_code, indicator_code, start_year, end_year)
    source_error = None
except Exception as exc:
    original = pd.DataFrame(columns=["year", "value"])
    source_error = str(exc)
    st.error(f"World Bank data could not be retrieved: {source_error}. Check network access/API status and retry. No substitute data has been generated.")

if not original.empty:
    original = original.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    st.write(f"**Actual coverage:** {int(original.year.min()) if len(original) else 'none'}–{int(original.year.max()) if len(original) else 'none'} · **Valid observations:** {len(original)} (not guaranteed to be 70).")
    st.dataframe(original.rename(columns={"year":"Year", "value":f"Value ({units})"}), use_container_width=True, hide_index=True)
else:
    st.warning("No valid non-null observations are available for the selected series.")

if "edited_data" not in st.session_state or st.session_state.get("data_key") != (country_code, indicator_code):
    st.session_state.edited_data = original.copy()
    st.session_state.data_key = (country_code, indicator_code)
    st.session_state.user_modified = False
if "calculated_years" not in st.session_state:
    st.session_state.calculated_years = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()

st.subheader("Editable working data")
st.caption("Edit the year/value table below. Source data above stays unchanged. Blank values are excluded from fitting; duplicate years are rejected.")
edited = st.data_editor(
    st.session_state.edited_data,
    num_rows="dynamic", use_container_width=True, key=f"editor_{country_code}_{indicator_code}",
    column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"), "value": st.column_config.NumberColumn(f"Value ({units})", format="%.8f")}
)
c1, c2 = st.columns([1, 4])
with c1:
    if st.button("Reset to source"):
        st.session_state.edited_data = original.copy()
        st.session_state.user_modified = False
        st.session_state.calculated_years = []
        st.session_state.batch_results = pd.DataFrame()
        st.rerun()
with c2:
    st.write("Status: **User-modified scenario**" if st.session_state.get("user_modified", False) else "**Source data / not marked modified**")

valid, validation_errors = validate_data(edited, indicator_name)
if validation_errors:
    for error in validation_errors:
        st.error(error)
    model_ready = False
else:
    model_ready = len(valid) >= degree + 1 and valid["year"].nunique() >= degree + 1
    if not model_ready:
        st.warning(f"Not enough distinct valid years for degree {degree}. Need at least {degree+1}; currently {valid.year.nunique()}.")
if not edited.reset_index(drop=True).equals(st.session_state.edited_data.reset_index(drop=True)):
    st.session_state.user_modified = True
st.session_state.edited_data = edited.copy()

base_year = None
coeff = None
fit_metrics = None
equation = "No fitted equation available."
holdout = None
if model_ready:
    try:
        base_year, coeff, fitted, fit_metrics = fit_poly(valid, degree)
        equation = poly_equation(coeff, base_year)
        holdout = fit_holdout(valid, degree)
    except Exception as exc:
        st.error(f"Model fit failed: {exc}")
        model_ready = False

# Fit/model summary and graph
st.subheader("Polynomial model and graph")
if model_ready:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Degree", degree)
    metric_cols[1].metric("R² (in-sample)", fmt_num(fit_metrics["r2"], 6))
    metric_cols[2].metric("MAE (in-sample)", f"{fmt_num(fit_metrics['mae'], 6)} {units}")
    metric_cols[3].metric("Observations", len(valid))
    metric_cols[4].metric("Mean", f"{fmt_num(valid.value.mean(), 6)} {units}")
    st.code(equation)
    st.caption(f"Centered time: t = year − {base_year}. Fit range: {valid.year.min()}–{valid.year.max()}. R² and MAE are in-sample fit metrics, not proof of causation or forecast accuracy.")
    if holdout:
        st.write(f"**Chronological holdout:** last {holdout['n']} observation(s) held out; MAE = {fmt_num(holdout['mae'], 6)} {units}. This is a separate test score, and the model was fit only on earlier years.")
    else:
        st.caption("Chronological holdout score unavailable: not enough observations remain after reserving training data.")
    residual_df = valid.copy()
    residual_df["fitted"] = fitted
    residual_df["residual (actual − fitted)"] = fit_metrics["residuals"]
    with st.expander("Fitted values and residuals"):
        st.dataframe(residual_df, use_container_width=True, hide_index=True)
    lo, hi = int(valid.year.min()), int(valid.year.max())
    plot_end = hi + horizon
    dense_years = np.linspace(lo, plot_end, max(250, (plot_end-lo+1)*3))
    dense_vals = np.polynomial.polynomial.polyval(dense_years-base_year, coeff)
    sampled_years = np.arange(lo, plot_end+1, sample_increment)
    sampled_vals = np.polynomial.polynomial.polyval(sampled_years-base_year, coeff)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(valid.year, valid.value, label="Observed data", zorder=4)
    ax.plot(dense_years[dense_years <= hi], dense_vals[dense_years <= hi], label="Fitted curve (historical range)")
    if plot_end > hi:
        ax.plot(dense_years[dense_years >= hi], dense_vals[dense_years >= hi], linestyle="--", label="Extrapolated segment")
    ax.axvline(hi, linestyle=":", label="Historical boundary")
    ax.scatter(sampled_years, sampled_vals, marker="x", s=25, label=f"Model markers every {sample_increment} year(s)")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.set_title(f"{country_name}: {indicator_name}")
    ax.grid(True, alpha=.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    if horizon:
        future_values = np.polynomial.polynomial.polyval(np.arange(hi+1, hi+horizon+1)-base_year, coeff)
        if len(future_values):
            scale = max(float(np.nanmax(np.abs(valid.value))), 1.0)
            if np.nanmax(np.abs(future_values)) > 10*scale or np.nanmin(future_values) < -10*scale:
                st.warning("The polynomial extrapolation becomes extreme relative to observed values. This is a warning, not a correction; predictions are not clipped.")
    if printer_mode:
        st.markdown("### Printer-friendly report")
        st.write(f"{country_name} · {indicator_name} · {valid.year.min()}–{valid.year.max()} · n={len(valid)}")
        st.write(equation)
        st.write(f"In-sample R²: {fmt_num(fit_metrics['r2'])}; MAE: {fmt_num(fit_metrics['mae'])} {units}.")
else:
    fig = None
    st.info("A fitted graph and model metrics will appear when the working table has enough valid, distinct years.")

# Single-year and batch calculations
st.subheader("Calculate fitted values")
single_year = st.number_input("Year to calculate (1950–2100)", min_value=1950, max_value=2100, value=min(max(end_year,1950),2100), step=1)
if st.button("Calculate Value", disabled=not model_ready):
    estimate = float(np.polynomial.polynomial.polyval(single_year-base_year, coeff))
    classification = "within fitted range" if valid.year.min() <= single_year <= valid.year.max() else "outside fitted range (extrapolation)"
    st.session_state.calculated_years = [r for r in st.session_state.calculated_years if r["year"] != int(single_year)]
    st.session_state.calculated_years.append({"year":int(single_year),"estimate":estimate,"classification":classification})
    st.session_state.calc_signature = (country_code, indicator_code, degree, tuple(valid.year), tuple(np.round(valid.value, 12)))
if st.session_state.get("calculated_years"):
    st.dataframe(pd.DataFrame(st.session_state.calculated_years), use_container_width=True, hide_index=True)
    if st.button("Clear calculated years"):
        st.session_state.calculated_years = []
        st.rerun()

st.markdown("**Batch years**")
batch_mode = st.radio("Batch input method", ["Year range", "Custom comma-separated years"], horizontal=True)
batch_years = []
if batch_mode == "Year range":
    b1, b2, b3 = st.columns(3)
    with b1: batch_start = st.number_input("Start year", 1950, 2100, 2000, 1)
    with b2: batch_end = st.number_input("End year", 1950, 2100, 2010, 1)
    with b3: batch_step = st.number_input("Step (1–10 years)", 1, 10, 1, 1)
    if batch_start <= batch_end:
        batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
    else:
        st.error("Batch range start must be less than or equal to end.")
else:
    custom_years_text = st.text_input("Years separated by commas", "2000, 2005, 2010")
    for part in custom_years_text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            year = int(part)
            if not 1950 <= year <= 2100:
                st.warning(f"Ignored {part}: years must be 1950–2100.")
            else:
                batch_years.append(year)
        except ValueError:
            st.warning(f"Ignored '{part}': enter whole-number years separated by commas.")
    batch_years = list(dict.fromkeys(batch_years))
if st.button("Calculate batch", disabled=not model_ready):
    rows = []
    for year in batch_years:
        estimate = float(np.polynomial.polynomial.polyval(year-base_year, coeff))
        rows.append({"year":year, "estimate":estimate, "units":units,
                     "classification":"within fitted range" if valid.year.min() <= year <= valid.year.max() else "outside fitted range (extrapolation)"})
    st.session_state.batch_results = pd.DataFrame(rows)
    st.session_state.calc_signature = (country_code, indicator_code, degree, tuple(valid.year), tuple(np.round(valid.value, 12)))
if not st.session_state.batch_results.empty:
    st.dataframe(st.session_state.batch_results, use_container_width=True, hide_index=True)

# Comparison series
if comparison_names:
    st.subheader("Compare countries")
    comp_fig, comp_ax = plt.subplots(figsize=(11, 4.5))
    comp_ax.plot(valid.year, valid.value, marker="o", label=f"{country_name} (working data)")
    comp_notes = []
    for name in comparison_names:
        code = COUNTRIES[name]
        try:
            comp = cached_wb_fetch(code, indicator_code, start_year, end_year).dropna(subset=["value"]).sort_values("year")
            if comp.empty:
                comp_notes.append(f"{name}: no non-null observations returned.")
                continue
            comp_ax.plot(comp.year, comp.value, marker=".", linewidth=1.2, label=name)
            comp_notes.append(f"{name}: coverage {int(comp.year.min())}–{int(comp.year.max())}, n={len(comp)}.")
        except Exception as exc:
            comp_notes.append(f"{name}: API retrieval failed ({exc}).")
    comp_ax.set_xlabel("Year")
    comp_ax.set_ylabel(f"{indicator_name} ({units})")
    comp_ax.set_title(f"Country comparison: {indicator_name}")
    comp_ax.grid(True, alpha=.25)
    comp_ax.legend()
    st.pyplot(comp_fig, use_container_width=True)
    st.caption("Country coverage can differ because data availability, reporting, and definitions vary. The graph shows observed values only; no missing values are invented.")
    for note in comp_notes:
        st.write("• " + note)

# General polynomial interpretation
if model_ready:
    with st.expander("Interpretation for your mathematics pathway", expanded=True):
        st.write(f"**Domain used for fitting:** observed years {int(valid.year.min())}–{int(valid.year.max())}. The polynomial formula is mathematically defined for all real t, but historical evidence only covers the displayed observed years.")
        st.write(f"**Range on observed years:** observed values run from {fmt_num(valid.value.min())} to {fmt_num(valid.value.max())} {units}; this observed range is not the full mathematical range of the polynomial.")
        derivative = np.polynomial.polynomial.polyder(coeff)
        second_derivative = np.polynomial.polynomial.polyder(derivative)
        st.write("**End behavior:** determined by the leading coefficient and degree; high-degree polynomial behavior far beyond the data can be unrealistic.")
        if len(valid) > 1:
            slopes = np.diff(valid.value.to_numpy()) / np.diff(valid.year.to_numpy())
            st.write(f"**Average rates between adjacent observations:** from {fmt_num(np.min(slopes))} to {fmt_num(np.max(slopes))} {units}/year.")
        if degree >= 3:
            st.write("**Precalculus extension:** degree changes the possible number of turns; coefficients control vertical scale, shifts, and curvature. Turning points should be checked from the derivative, not assumed from degree alone.")
        st.write("**Calculus extension:** derivatives of the centered polynomial are:")
        d1 = " + ".join(f"({c:.8g})·t^{i}" for i,c in enumerate(derivative))
        d2 = " + ".join(f"({c:.8g})·t^{i}" for i,c in enumerate(second_derivative))
        st.latex(r"f'(t)=" + sp.latex(sum(sp.Float(c, 8)*X**i for i,c in enumerate(derivative))))
        st.latex(r"f''(t)=" + sp.latex(sum(sp.Float(c, 8)*X**i for i,c in enumerate(second_derivative))))
        st.write("Numerical roots of the first derivative can suggest critical points; a point is classified only after checking the sign change. Endpoints must be included when finding absolute extrema on a closed interval.")

# Function lab / calculus simulator
st.header("Function Lab: contexts, functions, and calculus")
st.caption("This section is a mathematical simulation. Suggested functions and context interpretations are illustrative, not claims about historical causes or professional advice.")
course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
context = st.selectbox("Context", list(CONTEXTS))
default_y, default_yu, default_x, default_xu, context_note = CONTEXTS[context]
context_description = st.text_input("Context description", context_note)
var_y = st.text_input("Dependent variable name", default_y)
unit_y = st.text_input("Dependent variable units", default_yu)
var_x = st.text_input("Independent variable name", default_x)
unit_x = st.text_input("Independent variable units", default_xu)
function_source = st.radio("Function source", ["Suggested function", "Custom function", "Use fitted historical curve"], horizontal=True)
if function_source == "Suggested function":
    function_text = SUGGESTED[context]
    st.code(function_text)
    st.caption("Suggested function uses parameters a, b, c, and/or k. Set their values below.")
elif function_source == "Custom function":
    function_text = st.text_input("Enter f(x), e.g. 2*x**2 - 3*x + 1, exp(-0.2*x), sin(x)", "2*x**2 - 3*x + 1")
else:
    function_text = "historical_fit"
param_cols = st.columns(4)
params = {}
for i, (key, val) in enumerate([("a",1.0),("b",0.2),("c",0.0),("k",0.1)]):
    with param_cols[i]:
        params[key] = st.number_input(f"Parameter {key}", value=val, step=0.1, format="%.4f", key=f"param_{key}_{context}")
xmin, xmax = st.columns(2)
with xmin: interval_start = st.number_input("Analysis interval start", value=0.0, step=1.0)
with xmax: interval_end = st.number_input("Analysis interval end", value=10.0, step=1.0)
if interval_end <= interval_start:
    st.error("Analysis interval end must be greater than start.")
    interval_end = interval_start + 1
x0 = st.slider("Current input value x", min_value=float(interval_start), max_value=float(interval_end), value=float((interval_start+interval_end)/2), step=max((interval_end-interval_start)/100, .001))
r1, r2, r3 = st.columns(3)
with r1: avg_a = st.slider("Average-rate interval start", float(interval_start), float(interval_end), float(interval_start), step=max((interval_end-interval_start)/100,.001))
with r2: avg_b = st.slider("Average-rate interval end", float(interval_start), float(interval_end), float(interval_end), step=max((interval_end-interval_start)/100,.001))
with r3: show_f = st.checkbox("Show function", True); show_d1 = st.checkbox("Show first derivative", course=="Calculus"); show_d2 = st.checkbox("Show second derivative", course=="Calculus")
animate = st.checkbox("Automatic playback (updates on rerun)", value=False)
speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
if animate:
    st.caption(f"Playback requested: {speed}. Streamlit reruns on interaction; use the slider to advance the moving point. Automatic timed playback is not enabled in this dependency-light version.")
st.write(f"**Context meaning:** {context_description}")

expr = None
lab_error = None
lab_historical = function_source == "Use fitted historical curve"
if lab_historical:
    if not model_ready:
        lab_error = "Fit the historical data model first; otherwise the historical curve is unavailable."
else:
    try:
        expr = parse_expr(function_text)
        expr = expr.subs({sp.Symbol(k): sp.Float(v) for k,v in params.items()})
        expr = sp.simplify(expr)
    except Exception as exc:
        lab_error = str(exc)

if not lab_error:
    if lab_historical:
        def f_eval(arr):
            arr = np.asarray(arr, dtype=float)
            return np.polynomial.polynomial.polyval(arr-base_year, coeff)
        def d1_eval(arr):
            return np.polynomial.polynomial.polyval(np.asarray(arr)-base_year, np.polynomial.polynomial.polyder(coeff))
        def d2_eval(arr):
            return np.polynomial.polynomial.polyval(np.asarray(arr)-base_year, np.polynomial.polynomial.polyder(coeff, 2))
        f_label = equation
        domain_note = "Polynomial is defined for all real years; interpretation outside historical coverage is extrapolation."
    else:
        d1_expr = sp.diff(expr, X)
        d2_expr = sp.diff(expr, X, 2)
        f_eval = lambda arr: safe_eval(expr, np.asarray(arr, dtype=float))
        d1_eval = lambda arr: safe_eval(d1_expr, np.asarray(arr, dtype=float))
        d2_eval = lambda arr: safe_eval(d2_expr, np.asarray(arr, dtype=float))
        f_label = f"f(x) = {sp.sstr(expr)}"
        domain_note = "Domain restrictions are determined by denominators, logarithms, square roots, and other function definitions; undefined samples are left as gaps."
    st.latex(sp.latex(expr) if not lab_historical else r"f(x)=\text{centered historical polynomial}")
    if not lab_historical:
        st.write(f"**Function:** {f_label}")
        st.write(f"**First derivative:** {sp.sstr(d1_expr)}")
        st.write(f"**Second derivative:** {sp.sstr(d2_expr)}")
    try:
        grid = np.linspace(float(interval_start), float(interval_end), 900)
        ygrid = f_eval(grid)
        d1grid = d1_eval(grid)
        d2grid = d2_eval(grid)
        y0 = float(np.asarray(f_eval(np.array([x0]))).reshape(-1)[0])
        slope0 = float(np.asarray(d1_eval(np.array([x0]))).reshape(-1)[0])
        ya = float(np.asarray(f_eval(np.array([avg_a]))).reshape(-1)[0])
        yb = float(np.asarray(f_eval(np.array([avg_b]))).reshape(-1)[0])
        avg_rate = (yb-ya)/(avg_b-avg_a) if avg_b != avg_a and np.isfinite(ya) and np.isfinite(yb) else np.nan
        ffig, fax = plt.subplots(figsize=(10, 5.5))
        if show_f:
            fax.plot(grid, np.where(np.isfinite(ygrid), ygrid, np.nan), label="f(x)")
            fax.scatter([x0], [y0] if np.isfinite(y0) else [np.nan], label="Moving point", zorder=5)
            tangent_y = y0 + slope0*(grid-x0)
            if np.isfinite(y0) and np.isfinite(slope0):
                fax.plot(grid, tangent_y, linestyle="--", label="Tangent at current x")
            if np.isfinite(avg_rate) and avg_b != avg_a:
                secant_y = ya + avg_rate*(grid-avg_a)
                fax.plot(grid, secant_y, linestyle=":", label="Secant over selected interval")
        if show_d1: fax.plot(grid, d1grid, label="f′(x)")
        if show_d2: fax.plot(grid, d2grid, label="f″(x)")
        fax.axvline(x0, linestyle=":", alpha=.5)
        fax.set_xlabel(f"{var_x} ({unit_x})")
        fax.set_ylabel(f"{var_y} / derivative units")
        fax.set_title(f"{context}: function and rate of change")
        fax.grid(True, alpha=.25); fax.legend()
        st.pyplot(ffig, use_container_width=True)
        v1, v2, v3 = st.columns(3)
        v1.metric(f"{var_y} at x", fmt_num(y0, 8))
        v2.metric(f"Instantaneous rate ({unit_y}/{unit_x})", fmt_num(slope0, 8))
        v3.metric(f"Average rate ({unit_y}/{unit_x})", fmt_num(avg_rate, 8))
        st.write(f"**Interpretation:** At {var_x} = {x0:g} {unit_x}, the model gives {var_y} ≈ {fmt_num(y0,8)} {unit_y}; its instantaneous rate is ≈ {fmt_num(slope0,8)} {unit_y}/{unit_x}. Between {avg_a:g} and {avg_b:g} {unit_x}, the average rate is ≈ {fmt_num(avg_rate,8)} {unit_y}/{unit_x}.")
        if context == "Engineering":
            st.write(f"Here the first derivative represents velocity and the second derivative represents acceleration, with units {unit_y}/{unit_x} and {unit_y}/{unit_x}².")
        else:
            st.write(f"The second derivative is ≈ {fmt_num(float(np.asarray(d2_eval(np.array([x0]))).reshape(-1)[0]),8)} {unit_y}/{unit_x}² and describes how the rate of change is changing.")
        st.caption(domain_note)
        if course == "Calculus":
            st.markdown("#### Numerical calculus analysis")
            finite = np.isfinite(ygrid) & np.isfinite(d1grid) & np.isfinite(d2grid)
            # Approximate critical points by sign changes in sampled first derivative.
            crit_indices = np.where(np.diff(np.signbit(d1grid)))[0]
            critical = []
            for idx in crit_indices:
                xx = float((grid[idx]+grid[idx+1])/2)
                yy = float(np.asarray(f_eval(np.array([xx]))).reshape(-1)[0])
                left_sign = np.sign(d1grid[max(0,idx-2)])
                right_sign = np.sign(d1grid[min(len(d1grid)-1,idx+3)])
                classification = "local minimum" if left_sign < 0 < right_sign else ("local maximum" if left_sign > 0 > right_sign else "stationary; no extremum confirmed")
                critical.append({"x approx.":xx, "f(x) approx.":yy, "classification":classification})
            if critical:
                st.dataframe(pd.DataFrame(critical), hide_index=True, use_container_width=True)
            else:
                st.write("No derivative sign-change critical points detected on this sampled interval. This does not prove none exist.")
            if finite.sum() > 5:
                vals = ygrid[finite]; xs = grid[finite]
                imin, imax = int(np.argmin(vals)), int(np.argmax(vals))
                st.write(f"**Approximate absolute extrema on selected interval (sampled):** minimum ≈ {fmt_num(vals[imin])} at x ≈ {xs[imin]:.5g}; maximum ≈ {fmt_num(vals[imax])} at x ≈ {xs[imax]:.5g}. Endpoints are included in the grid.")
                inflect_idx = np.where(np.diff(np.signbit(d2grid)))[0]
                inflect = []
                for idx in inflect_idx:
                    xx = float((grid[idx]+grid[idx+1])/2)
                    inflect.append({"x approx.":xx,"f(x) approx.":float(np.asarray(f_eval(np.array([xx]))).reshape(-1)[0]),"note":"second-derivative sign change sampled"})
                st.write("**Possible inflection points:** only points where concavity appears to change are listed; numerical approximations need verification.")
                if inflect: st.dataframe(pd.DataFrame(inflect), hide_index=True, use_container_width=True)
                else: st.write("No sampled concavity changes detected.")
                increasing = np.diff(ygrid) > 0
                st.write(f"**Sampled trend:** {int(np.sum(increasing))} of {len(increasing)} adjacent grid steps increase. Numerical sampling is approximate.")
        elif course == "Precalculus":
            st.write("**Turning points and concavity:** inspect the graph and derivative curves. Degree and coefficients influence turning behavior and curvature; numerical graphing suggests, but does not prove, exact critical points.")
        else:
            st.write("**Algebra 2 focus:** compare function values, slopes, domain/range over the selected interval, and how the graph behaves toward each end of the interval.")
        # Simple contextual visual tied to model value, normalized only for display.
        st.markdown("#### Contextual visual simulation")
        valid_y = ygrid[np.isfinite(ygrid)]
        if len(valid_y):
            y_min, y_max = float(np.min(valid_y)), float(np.max(valid_y))
            fraction = .5 if y_max == y_min or not np.isfinite(y0) else float(np.clip((y0-y_min)/(y_max-y_min),0,1))
            if context in ("Engineering","Sports science"):
                st.progress(fraction, text=f"Moving-object state (normalized to this graph's interval): {fraction:.0%}")
            elif context in ("Medical","Bioengineering"):
                st.progress(fraction, text=f"Signal/concentration indicator (normalized to this graph's interval): {fraction:.0%}")
            elif context == "Biology":
                st.progress(fraction, text=f"Population indicator (normalized to this graph's interval): {fraction:.0%}")
            elif context == "Weather":
                st.progress(fraction, text=f"Temperature indicator (normalized to this graph's interval): {fraction:.0%}")
            else:
                st.progress(fraction, text=f"Context indicator (normalized to this graph's interval): {fraction:.0%}")
            st.caption("The visual scales the selected function's current value between its finite graph minimum and maximum; it is a display aid, not a real physical simulation.")
    except Exception as exc:
        st.error(f"Function evaluation failed. Check the function, parameters, and interval. Details: {exc}")
else:
    st.error(lab_error)

# Exports
st.header("Export current work")
st.caption("Exports include current selection, source metadata, working data, fit metrics when available, and calculated results.")
export_results = pd.DataFrame(st.session_state.calculated_years)
if not st.session_state.batch_results.empty:
    export_results = pd.concat([export_results, st.session_state.batch_results], ignore_index=True, sort=False)
summary = pd.DataFrame([{
    "country":country_name, "country_code":country_code, "indicator":indicator_name, "indicator_code":indicator_code,
    "units":units, "definition":definition, "requested_start_year":start_year, "requested_end_year":end_year,
    "actual_start_year":int(valid.year.min()) if len(valid) else None, "actual_end_year":int(valid.year.max()) if len(valid) else None,
    "observation_count":len(valid), "degree":degree if model_ready else None, "base_year":base_year,
    "equation":equation, "r_squared_in_sample":fit_metrics["r2"] if model_ready else None,
    "mae_in_sample":fit_metrics["mae"] if model_ready else None,
    "data_status":"user-modified scenario" if st.session_state.get("user_modified",False) else "source data",
    "source_url":worldbank_link(country_code,indicator_code), "retrieval_date":date.today().isoformat()
}])
csv_bytes = valid.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    valid.to_excel(writer, sheet_name="Working Data", index=False)
    export_results.to_excel(writer, sheet_name="Calculated Years", index=False)
    if model_ready:
        residual_df.to_excel(writer, sheet_name="Fit and Residuals", index=False)
st.download_button("Download Excel workbook (.xlsx)", xlsx_buffer.getvalue(), file_name="hispanic_heritage_data_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if model_ready:
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=140, bbox_inches="tight")
    chart_png = chart_buffer.getvalue()
    metric_lines = [
        f"Observations: {len(valid)}; fitted range: {valid.year.min()}–{valid.year.max()}; degree: {degree}.",
        f"In-sample R-squared: {fmt_num(fit_metrics['r2'])}; in-sample MAE: {fmt_num(fit_metrics['mae'])} {units}.",
        f"Mean observed value: {fmt_num(valid.value.mean())} {units}.",
        f"Data status: {'user-modified scenario' if st.session_state.get('user_modified',False) else 'source data'}."
    ]
    findings = [
        "Polynomial models describe patterns in the selected data; they do not establish historical causes.",
        "MAE and residuals describe in-sample fit error. They are not validated 95% prediction intervals.",
        "Extrapolation can be unstable, especially for higher-degree polynomials. Predictions are not clipped.",
        f"Source: World Bank API indicator {indicator_code}. Coverage can be shorter than the requested period."
    ]
    pdf_bytes = build_pdf("Hispanic Heritage Historical Data Explorer",
        [f"Country: {country_name} ({country_code})", f"Indicator: {indicator_name} ({indicator_code})",
         f"Units: {units}", f"Retrieved: {date.today().isoformat()}", f"Status: {'user-modified scenario' if st.session_state.get('user_modified',False) else 'source data'}"],
        equation, metric_lines, chart_png, valid, export_results, findings)
    st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_data_explorer.pdf", mime="application/pdf")
else:
    st.info("PDF report becomes available after a valid model fit. CSV and Excel remain available.")

st.markdown("---")
st.markdown("**Data flow:** World Bank API → cached source observations → editable working table → validation → centered-time polynomial fit → estimates/plots/exports. The Function Lab is a separate teaching simulator unless “Use fitted historical curve” is selected.")
st.markdown("**Limitations:** API coverage may be incomplete; missing observations are not fabricated. High-degree polynomial fits can overfit. R²/MAE are descriptive, not causal. Numerical derivative analysis uses sampling and may miss closely spaced roots or discontinuities. Function Lab outputs are educational approximations, not professional, medical, financial, engineering, or weather advice.")
