from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

st.set_page_config(page_title="Task 9 | Titanic Statistics", page_icon="🚢", layout="wide")
st.title("🚢 Task 9 — Titanic Descriptive Statistics")
st.caption("Python • Pandas • Excel | Mean, median, mode, standard deviation and percentiles")

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return pd.read_csv(DATA_URL)

try:
    df = load_data()
except Exception as exc:
    st.error("Dataset could not be loaded. Run `python analysis.py` first, or check your internet connection.")
    st.exception(exc)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Passenger records", f"{len(df):,}")
c2.metric("Columns", str(df.shape[1]))
c3.metric("Survival rate", f"{df['survived'].mean()*100:.1f}%")
c4.metric("Missing age values", f"{df['age'].isna().sum():,}")

st.subheader("Choose numeric columns")
numeric_options = [c for c in ["survived", "pclass", "age", "sibsp", "parch", "fare"] if c in df.columns]
selected = st.multiselect("Columns to summarize", numeric_options, default=["age", "fare", "survived"])
if selected:
    rows = []
    for col in selected:
        s = df[col].dropna()
        modes = s.mode()
        rows.append({
            "Column": col, "Count": int(s.count()), "Missing": int(df[col].isna().sum()),
            "Mean": s.mean(), "Median": s.median(),
            "Mode": ", ".join(map(str, modes.tolist())),
            "Standard deviation": s.std(), "Min": s.min(),
            "25th percentile": s.quantile(.25), "50th percentile": s.quantile(.5),
            "75th percentile": s.quantile(.75), "Max": s.max(),
        })
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Distribution")
    col = st.selectbox("Select a column for histogram", ["age", "fare", "sibsp", "parch"])
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=25, edgecolor="black")
    ax.set_xlabel(col.title())
    ax.set_ylabel("Passenger count")
    ax.set_title(f"Distribution of {col.title()}")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Survival by passenger class")
    grouped = df.groupby("pclass")["survived"].mean().mul(100)
    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Survival rate (%)")
    ax.set_title("Survival rate by class")
    ax.set_xticklabels([str(int(x)) for x in grouped.index], rotation=0)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Missing-value check")
missing = pd.DataFrame({"Missing values": df.isna().sum(), "Missing %": (df.isna().mean()*100).round(2)})
st.dataframe(missing, use_container_width=True)

st.subheader("Dataset preview")
st.dataframe(df.head(15), use_container_width=True)
st.info("Interpretation tip: compare mean with median to spot skew. Use standard deviation with the mean, and the 25th/75th percentiles to understand the middle 50% of values.")
