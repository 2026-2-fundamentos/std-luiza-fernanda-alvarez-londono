import re
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ACTIVITY_DIR / "data" / "ventas.csv"
OUTPUT_FILE = ACTIVITY_DIR / "submission" / "ventas.csv"

SUPPLIER_REPLACEMENTS = {
    "amazon web services colombia": "Amazon Web Services Colombia",
    "BANCOLOMBIA S.A.": "Bancolombia S.A.",
    "Cementos Argos SA": "Cementos Argos S.A.",
    "cementos argos s.a.": "Cementos Argos S.A.",
    "Corona SAS": "Corona S.A.S.",
    "Ecopetrol  S.A.": "Ecopetrol S.A.",
    "GOOGLE COLOMBIA LTDA.": "Google Colombia Ltda.",
    "Grupo  Éxito  S.A.": "Grupo Éxito S.A.",
    "IBM Colombia SAS.": "IBM Colombia S.A.S.",
    "ibm colombia s.a.s.": "IBM Colombia S.A.S.",
    "MICROSOFT COLOMBIA INC.": "Microsoft Colombia Inc.",
    "Nutresa SA": "Nutresa S.A.",
    "nutresa s.a.": "Nutresa S.A.",
    "oracle colombia ltda.": "Oracle Colombia Ltda.",
    "POSTOBÓN S.A.": "Postobon S.A.",
    "SAP Colombia SAS": "SAP Colombia S.A.S.",
    "Schneider  Electric": "Schneider Electric",
    "SIEMENS S.A.S.": "Siemens S.A.S.",
    "siemens s.a.s.": "Siemens S.A.S.",
    "Sura  S.A.": "Sura S.A.",
    "Telefónica  Colombia": "Telefónica Colombia",
}


COUNTRY_REPLACEMENTS = {
    "CO": "COL",
    "COLOMBIA": "COL",
    "Colombia": "COL",
    "colombia": "COL",
}

CITY_REPLACEMENTS = {
    "bogotá": "Bogotá",
    "BOGOTÁ": "Bogotá",
    "Medellin": "Medellín",
    "medellín": "Medellín",
    "MEDELLÍN": "Medellín",
}

# Factores para convertir el peso a kilogramos
WEIGHT_FACTORS = {
    "": 1,  # sin unidad: se asume kilogramos
    "kg": 1,
    "kgs": 1,
    "g": 0.001,
    "gr": 0.001,
    "lb": 0.453592,
    "lbs": 0.453592,
    "ton": 1000,
    "t": 1000,
}


def clean_column_names(df):
    # Casos tipicos en la columna
    # ' Supplier ID ' -> "supplier_id"
    # 'City ' -> "city"
    # 'Purchase Date' -> "purchase_date"
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_supplier_id_column(df):
    # ' 0007 ' -> '0007'
    df["supplier_id"] = df["supplier_id"].astype(str).str.strip().str.zfill(4)
    return df


def clean_supplier_column(df):
    df["supplier"] = df["supplier"].str.strip()
    df["supplier"] = df["supplier"].replace(SUPPLIER_REPLACEMENTS)
    return df


def clean_country_column(df):
    df["country"] = df["country"].str.strip()
    df["country"] = df["country"].replace(COUNTRY_REPLACEMENTS)
    return df


def clean_city_column(df):
    df["city"] = df["city"].str.strip()
    df["city"] = df["city"].replace(CITY_REPLACEMENTS)
    return df


def clean_purchase_date_column(df):

    def convert_yyyy_s_mm_s_dd_to_yyyy_mm_dd(series):
        # Convierte fechas en formato "YYYY/MM/DD" a "YYYY-MM-DD"
        series = series.str.replace(r"(\d{4})/(\d{2})/(\d{2})", r"\1-\2-\3", regex=True)
        return series

    def convert_mmddyyyy_to_yyyy_mm_dd(series):
        # Corrige solo fechas con el día primero, cuando el día es mayor que 12.
        def convert_date(match):
            g1, g2, g3 = match.groups()
            if int(g2) > 12:  # el dia aparece en la mitad
                return f"{g3}-{g1}-{g2}"
            return f"{g3}-{g2}-{g1}"

        return series.str.replace(r"(\d{2})/(\d{2})/(\d{4})", convert_date, regex=True)

    def convert_dd_mm_yyyy_to_yyyy_mm_dd_separated_with_dots(series):
        # Corrige fechas en formato "DD.MM.YYYY" a "YYYY-MM-DD"
        return series.str.replace(r"(\d{2})\.(\d{2})\.(\d{4})", r"\3-\2-\1", regex=True)

    def convert_dd_mm_yy_to_yyyy_mm_dd_separated_with_hyphens(series):
        # Corrige fechas en formato "DD-MM-YY" a "YYYY-MM-DD"
        return series.str.replace(
            r"^(\d{2})-(\d{2})-(\d{2})$", r"20\3-\2-\1", regex=True
        )

    df["purchase_date"] = df["purchase_date"].str.strip()
    df["purchase_date"] = convert_yyyy_s_mm_s_dd_to_yyyy_mm_dd(df["purchase_date"])
    df["purchase_date"] = convert_mmddyyyy_to_yyyy_mm_dd(df["purchase_date"])
    df["purchase_date"] = convert_dd_mm_yyyy_to_yyyy_mm_dd_separated_with_dots(
        df["purchase_date"]
    )
    df["purchase_date"] = convert_dd_mm_yy_to_yyyy_mm_dd_separated_with_hyphens(
        df["purchase_date"]
    )

    return df


def clean_amount_column(df):

    def clean_strings_ending_with_k(series):
        # Convierte valores como "1.5K" a "1500"
        def convert_value(value):
            if isinstance(value, str) and value.endswith("K"):
                try:
                    return int(float(value[:-1]) * 1000)
                except ValueError:
                    return value
            return value

        return series.apply(convert_value)

    def clean_strings_starting_with_COP(series):
        # Convierte valores como "COP 1,250,000" a "1250000"
        def convert_value(value):
            if isinstance(value, str) and value.startswith("COP"):
                try:
                    return int(value[3:].replace(",", "").strip())
                except ValueError:
                    return value
            return value

        return series.apply(convert_value)

    def clean_strings_starting_with_dollar(series):
        # Convierte valores como "$1,250,000" a "1250000"
        def convert_value(value):
            if isinstance(value, str) and value.startswith("$"):
                try:
                    return int(value[1:].replace(",", "").strip())
                except ValueError:
                    return value
            return value

        return series.apply(convert_value)

    def remove_dots(series):
        # Elimina los puntos de los valores numéricos
        def convert_value(value):
            if isinstance(value, str):
                try:
                    return int(value.replace(".", "").strip())
                except ValueError:
                    return value
            return value

        return series.apply(convert_value)

    df["amount"] = df["amount"].str.strip()
    df["amount"] = clean_strings_ending_with_k(df["amount"])
    df["amount"] = clean_strings_starting_with_COP(df["amount"])
    df["amount"] = clean_strings_starting_with_dollar(df["amount"])
    df["amount"] = remove_dots(df["amount"])
    # Deja la columna numérica (lo que no se pueda interpretar queda vacío)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round().astype("Int64")

    return df


def clean_discount_column(df):
    # Limpia la columna de descuento
    # " 20% " -> 0.2 ; "5" -> 0.05 ; "0.05" -> 0.05
    df["discount"] = df["discount"].str.strip()
    df["discount"] = df["discount"].str.replace("%", "", regex=False)
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")
    df["discount"] = df["discount"].map(
        lambda x: x / 100 if not pd.isna(x) and x > 1 else x
    )

    return df


def clean_weight_column(df):
    def convert_value(value):
        if pd.isna(value):
            return np.nan

        text = str(value).strip().lower()
        text = text.replace(" ", "").replace(",", ".")

        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([a-z]*)", text)

        if match is None:
            return np.nan

        number = float(match.group(1))
        unit = match.group(2)

        if unit not in WEIGHT_FACTORS:
            return np.nan

        number = number * WEIGHT_FACTORS[unit]

        if number < 0:
            return np.nan

        return number

    df["weight"] = df["weight"].apply(convert_value)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    return df

    df["weight"] = df["weight"].apply(convert_value)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    return df


def clean_units_column(df):
    # Unidades como enteros; "N/A" y valores negativos quedan vacíos
    df["units"] = pd.to_numeric(df["units"], errors="coerce").round().astype("Int64")
    df["units"] = df["units"].where(df["units"] >= 0)

    return df


def clean_unit_price_column(df):
    # Convierte valores como "96000.00", "980.000.00", "$125,000.00" o
    # "$-1,750,000.00" a número. El signo negativo se descarta porque el valor
    # coincide con amount / units (por ejemplo, 3500000 / 2 = 1750000).

    def convert_value(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace("$", "").replace("COP", "").replace(" ", "")
        text = text.lstrip("-")
        if "," in text:
            # Formato "1,250,000.00": la coma separa miles
            text = text.replace(",", "")
        elif text.count(".") > 1:
            # Formato "1.250.000.00": los puntos separan miles y el último es decimal
            integer_part, decimal_part = text.rsplit(".", 1)
            text = integer_part.replace(".", "") + "." + decimal_part
        try:
            return float(text)
        except ValueError:
            return np.nan

    df["unit_price"] = df["unit_price"].map(convert_value)

    return df


def clean_contact_email_column(df):
    # Normaliza los correos; "sin correo" y valores sin "@" quedan vacíos
    df["contact_email"] = df["contact_email"].str.strip().str.lower()
    df["contact_email"] = df["contact_email"].where(
        df["contact_email"].str.contains("@", na=False)
    )

    return df


def main():
    df = pd.read_csv(INPUT_FILE)
    df = clean_column_names(df)
    df = clean_supplier_id_column(df)
    df = clean_supplier_column(df)
    df = clean_country_column(df)
    df = clean_city_column(df)
    df = clean_purchase_date_column(df)
    df = clean_amount_column(df)
    df = clean_discount_column(df)
    df = clean_weight_column(df)
    df = clean_units_column(df)
    df = clean_unit_price_column(df)
    df = clean_contact_email_column(df)
    # Elimina filas repetidas (ej. 0005, 0010 y 0019 aparecen dos veces)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    

if __name__ == "__main__":
    main()