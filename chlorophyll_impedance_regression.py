from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


TSV_FILE = Path("Gala Refrence dataset_2023_Sundus.Riaz.xlsx - Sheet1.tsv")
XLSX_FILE = Path("Gala Refrence dataset_2023_Sundus.Riaz.xlsx")
RANDOM_STATE = 42
OUTLIER_Z_THRESHOLD = 8.0
TARGET_COL = "Chlorophyll  (IAD)"
OUTPUT_DIR = Path("outputs")


def normalize_week(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"week\s*(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    week = int(match.group(1))
    if week >= 5:
        return "Week 5+"
    return f"Week {week}"


def sample_number(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def parse_args() -> argparse.Namespace:
    normalized_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("filterindex0="):
            normalized_args.extend(["--filterindex0", arg.split("=", 1)[1]])
        else:
            normalized_args.append(arg)

    parser = argparse.ArgumentParser(description="Regresion de clorofila usando solo impedancia base.")
    parser.add_argument(
        "--filterindex0",
        default=None,
        help="Filtro opcional sobre la primera columna del fichero, Location/Orchard.",
    )
    return parser.parse_args(normalized_args)


def to_numeric_decimal_comma(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(frame, pd.Series):
        return pd.to_numeric(frame.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return frame.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce"))


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    mad[mad == 0] = np.nan
    z = 0.6745 * (values - med) / mad
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def read_source() -> pd.DataFrame:
    if TSV_FILE.exists():
        print(f"Leyendo datos desde TSV: {TSV_FILE}")
        return pd.read_csv(TSV_FILE, sep="\t")
    if XLSX_FILE.exists():
        print(f"Leyendo datos desde Excel: {XLSX_FILE}")
        return pd.read_excel(XLSX_FILE, sheet_name="Sheet1")
    raise FileNotFoundError(f"No encuentro {TSV_FILE} ni {XLSX_FILE}")


def load_and_clean(filterindex0: str | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = read_source()

    mag_cols = list(df.columns[5:205])
    phase_cols = list(df.columns[205:405])
    feature_cols = mag_cols + phase_cols

    data = df[df["Sample No."].notna()].copy()
    if filterindex0:
        available = sorted(data.iloc[:, 0].dropna().astype(str).str.strip().unique())
        data = data[data.iloc[:, 0].astype(str).str.strip() == filterindex0.strip()].copy()
        print(f"Filtro index0 / {df.columns[0]}: {filterindex0}")
        print(f"Valores disponibles en index0: {available}")
        print(f"Filas tras filtro index0: {len(data)}")

    data["index0"] = data.iloc[:, 0].astype(str).str.strip()
    data["week"] = data["Harvest at "].map(normalize_week)
    data["sample_number"] = data["Sample No."].map(sample_number)
    data[feature_cols] = to_numeric_decimal_comma(data[feature_cols])
    data[TARGET_COL] = to_numeric_decimal_comma(data[TARGET_COL])

    initial_rows = len(data)
    valid_label = data["week"].isin(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5+"])
    valid_sample = data["sample_number"].notna()
    valid_target = np.isfinite(data[TARGET_COL].to_numpy(dtype=float))
    valid_spectra = np.isfinite(data[feature_cols].to_numpy(dtype=float)).all(axis=1)
    positive_magnitude = (data[mag_cols].to_numpy(dtype=float) > 0).all(axis=1)

    base_keep = valid_label & valid_sample & valid_target & valid_spectra & positive_magnitude
    removed_basic = data.loc[~base_keep, ["index0", "Harvest at ", "Sample No.", "week", TARGET_COL]].copy()
    clean = data.loc[base_keep].copy()

    mag = np.log10(clean[mag_cols].to_numpy(dtype=float))
    phase = clean[phase_cols].to_numpy(dtype=float)
    summary = np.column_stack(
        [
            np.median(mag, axis=1),
            np.std(mag, axis=1),
            mag[:, 0] - mag[:, -1],
            np.median(phase, axis=1),
            np.std(phase, axis=1),
            phase[:, 0] - phase[:, -1],
        ]
    )
    clean["spectral_outlier_z"] = np.max(np.abs(robust_z(summary)), axis=1)
    removed_outliers = clean.loc[
        clean["spectral_outlier_z"] > OUTLIER_Z_THRESHOLD,
        ["index0", "Harvest at ", "Sample No.", "week", TARGET_COL, "spectral_outlier_z"],
    ].copy()
    clean = clean.loc[clean["spectral_outlier_z"] <= OUTLIER_Z_THRESHOLD].copy()

    mag = np.log10(clean[mag_cols].to_numpy(dtype=float))
    phase = clean[phase_cols].to_numpy(dtype=float)
    x_flat = np.hstack([mag, phase])
    y = clean[TARGET_COL].to_numpy(dtype=float)

    print("=" * 80)
    print("LIMPIEZA REGRESION CHLOROPHYLL")
    print("=" * 80)
    print(f"Filas iniciales con Sample No.: {initial_rows}")
    print(f"Eliminadas por etiqueta/muestra/espectro/target invalido: {len(removed_basic)}")
    if len(removed_basic):
        print(removed_basic.head(60).to_string(index=False))
        if len(removed_basic) > 60:
            print(f"... {len(removed_basic) - 60} filas mas omitidas en consola")
    print(f"Eliminadas como outlier espectral robusto (z > {OUTLIER_Z_THRESHOLD}): {len(removed_outliers)}")
    if len(removed_outliers):
        print(removed_outliers.sort_values("spectral_outlier_z", ascending=False).to_string(index=False))
    print(f"Filas finales para modelado: {len(clean)}")
    print("Distribucion por clase temporal usada solo para describir el dataset:")
    print(clean["week"].value_counts().sort_index().to_string())
    print("Resumen target Chlorophyll (IAD):")
    print(clean[TARGET_COL].describe().to_string())
    print()

    if len(clean) == 0:
        raise ValueError("No quedan filas para modelar.")

    return clean, x_flat, y


def print_regression_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print()
    preview = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "error": y_pred - y_true})
    print("Primeras predicciones del test:")
    print(preview.head(20).round(4).to_string(index=False))
    print()


def save_predictions(records: list[pd.DataFrame], output_path: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.concat(records, ignore_index=True).to_csv(output_path, index=False)
    print(f"Archivo guardado: {output_path}")


def output_suffix(filterindex0: str | None) -> str:
    if not filterindex0:
        return "all"
    return re.sub(r"[^a-z0-9]+", "_", filterindex0.lower()).strip("_")


def main() -> None:
    args = parse_args()
    clean, x_flat, y = load_and_clean(filterindex0=args.filterindex0)
    groups = (clean["index0"].astype(str) + "::" + clean["sample_number"].astype(str)).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    print(f"GroupKFold con {n_splits} folds, shuffle=True, agrupado por index0 + Sample No.")
    print("Entrada: espectro base [log10(magnitud), fase].")
    print("Salida: Chlorophyll (IAD), normalizada durante entrenamiento y devuelta a escala original.")
    print("No se usa Week como feature; solo se muestra para control del dataset.")
    print()

    models = {
        "KNN_REG": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "SVR_REG": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(C=3.0, kernel="rbf", gamma="scale", epsilon=0.03)),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "XGBOOST_REG": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.03,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        ),
    }

    prediction_records = []
    for name, model in models.items():
        pred = cross_val_predict(model, x_flat, y, groups=groups, cv=cv)
        print_regression_metrics(name, y, pred)
        prediction_records.append(
            pd.DataFrame(
                {
                    "model": name,
                    "index0": clean["index0"].to_numpy(),
                    "sample_number": clean["sample_number"].to_numpy(),
                    "week": clean["week"].to_numpy(),
                    "y_true": y,
                    "y_pred": pred,
                    "error": pred - y,
                }
            )
        )

    save_predictions(
        prediction_records,
        OUTPUT_DIR / f"chlorophyll_regression_predictions_vs_gt_{output_suffix(args.filterindex0)}.csv",
    )


if __name__ == "__main__":
    main()
