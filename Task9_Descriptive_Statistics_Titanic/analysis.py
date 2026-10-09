from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
DATA_FILE = DATA_DIR / "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

NUMERIC_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
COLUMN_DESCRIPTIONS = {
    "survived": "Survival flag (0 = did not survive, 1 = survived)",
    "pclass": "Passenger ticket class (1 = first, 2 = second, 3 = third)",
    "age": "Passenger age in years",
    "sibsp": "Siblings/spouses aboard",
    "parch": "Parents/children aboard",
    "fare": "Passenger fare",
}

def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print("Downloading the public Titanic dataset...")
        try:
            with urlopen(DATA_URL, timeout=30) as response:
                DATA_FILE.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the dataset. Check internet or manually save the CSV to {DATA_FILE}. "
                f"Source: {DATA_URL}. Details: {exc}"
            ) from exc

def make_summary(df):
    rows = []
    for col in NUMERIC_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce")
        modes = s.mode(dropna=True)
        rows.append({
            "column": col,
            "description": COLUMN_DESCRIPTIONS[col],
            "count": int(s.count()),
            "missing": int(s.isna().sum()),
            "mean": s.mean(),
            "median": s.median(),
            "mode": ", ".join(str(x) for x in modes.tolist()) if len(modes) else "N/A",
            "standard_deviation": s.std(),
            "minimum": s.min(),
            "25th_percentile": s.quantile(0.25),
            "50th_percentile": s.quantile(0.50),
            "75th_percentile": s.quantile(0.75),
            "maximum": s.max(),
            "IQR": s.quantile(0.75) - s.quantile(0.25),
        })
    return pd.DataFrame(rows)

def build_interpretation(df, summary):
    age = df["age"].dropna()
    fare = df["fare"].dropna()
    survival_rate = df["survived"].mean() * 100
    age_row = summary.loc[summary["column"] == "age"].iloc[0]
    fare_row = summary.loc[summary["column"] == "fare"].iloc[0]
    lines = [
        "# Task 9 — Interpretation Report",
        "",
        "## Dataset overview",
        f"- Number of passenger records: **{len(df):,}**",
        f"- Number of columns: **{df.shape[1]}**",
        f"- Overall survival rate in this dataset: **{survival_rate:.1f}%**",
        "",
        "## Descriptive-statistics interpretation",
        f"- **Age:** mean = {age.mean():.2f} years; median = {age.median():.2f} years; standard deviation = {age.std():.2f} years. "
        + ("The mean is above the median, suggesting a right-skewed age distribution." if age.mean() > age.median() else
           "The mean is below the median, suggesting a left-skewed age distribution." if age.mean() < age.median() else
           "The mean and median are equal, so these two measures alone do not indicate skew.")
        ,
        f"- **Fare:** mean = {fare.mean():.2f}; median = {fare.median():.2f}; standard deviation = {fare.std():.2f}. "
        + ("The mean is much higher than the median, indicating high-fare values pull the average upward." if fare.mean() > fare.median() * 1.2 else
           "Compare the mean and median to assess whether unusually high or low fares affect the average."),
        f"- **Age middle 50%:** from {age.quantile(.25):.2f} to {age.quantile(.75):.2f} years (IQR = {age.quantile(.75)-age.quantile(.25):.2f}).",
        f"- **Fare middle 50%:** from {fare.quantile(.25):.2f} to {fare.quantile(.75):.2f} (IQR = {fare.quantile(.75)-fare.quantile(.25):.2f}).",
        "- **Standard deviation:** a larger standard deviation means values are more spread out around the mean. It should be interpreted alongside the mean and the units.",
        "- **Median vs. mean:** the median is usually more representative when the distribution is skewed or contains extreme values.",
        "",
        "## Group comparison: survival by passenger class",
        "",
    ]
    group = df.groupby("pclass", dropna=False)["survived"].agg(["count", "mean"]).reset_index()
    for _, row in group.iterrows():
        lines.append(f"- Class {int(row['pclass'])}: {int(row['count'])} passengers; survival rate = {row['mean']*100:.1f}%.")
    lines += [
        "",
        "## Important note",
        "These are descriptive associations in the supplied dataset. They do not establish causation. Missing values are excluded from each statistic by Pandas; the cleaned export uses median imputation only for numeric columns so the original raw data remains available.",
        "",
    ]
    return "\n".join(lines)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dataset()
    df = pd.read_csv(DATA_FILE)
    missing_columns = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    summary = make_summary(df)
    missing = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_missing_count": [int(df[c].notna().sum()) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [float(df[c].isna().mean() * 100) for c in df.columns],
    })
    cleaned = df.copy()
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        if cleaned[col].notna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    group_survival = df.groupby(["pclass", "sex"], dropna=False)["survived"].agg(
        passenger_count="count", survival_rate="mean"
    ).reset_index()
    group_survival["survival_rate_percent"] = group_survival["survival_rate"] * 100

    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_titanic.csv", index=False)
    report = build_interpretation(df, summary)
    (REPORT_DIR / "interpretation_report.md").write_text(report, encoding="utf-8")

    workbook = OUTPUT_DIR / "titanic_descriptive_statistics.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary Statistics", index=False)
        missing.to_excel(writer, sheet_name="Missing Values", index=False)
        group_survival.to_excel(writer, sheet_name="Survival by Group", index=False)
        df.head(20).to_excel(writer, sheet_name="First 20 Rows", index=False)
        cleaned.to_excel(writer, sheet_name="Cleaned Data", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_len = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 42)
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

    print(f"Dataset shape: {df.shape}")
    print("\nSummary statistics:")
    print(summary[["column", "count", "missing", "mean", "median", "mode", "standard_deviation", "25th_percentile", "75th_percentile"]].to_string(index=False))
    print(f"\nExcel workbook created: {workbook}")
    print(f"CSV summary created: {OUTPUT_DIR / 'summary_statistics.csv'}")
    print(f"Interpretation report created: {REPORT_DIR / 'interpretation_report.md'}")

if __name__ == "__main__":
    main()
