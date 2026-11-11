from pathlib import Path

import pandas as pd

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ACTIVITY_DIR / "data" / "ventas.csv"
OUTPUT_FILE = ACTIVITY_DIR / "submission" / "ventas.csv"


def main():
    df = pd.read_csv(OUTPUT_FILE)
    series = df["purchase_date"]

    # series = series.replace(r"\d", "d", regex=True)

    # series = series[series.str.match(r"\d{2}/\d{2}/\d{4}")]

    series = series.drop_duplicates()
    series = series.sort_values(ascending=True)
    series = series.to_list()
    series = sorted(series, key=lambda x: len(x))

    for value in series:
        print(repr(value))


if __name__ == "__main__":
    main()