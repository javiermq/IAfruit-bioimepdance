from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


TSV_FILE = Path("Gala Refrence dataset_2023_Sundus.Riaz.xlsx - Sheet1.tsv")
XLSX_FILE = Path("Gala Refrence dataset_2023_Sundus.Riaz.xlsx")
RANDOM_STATE = 42
OUTLIER_Z_THRESHOLD = 8.0
OUTPUT_DIR = Path("outputs")


def normalize_week(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"week\s*(\d+)", text, flags=re.IGNORECASE)
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

    parser = argparse.ArgumentParser(
        description="Clasificacion de semana usando diferencia de impedancia respecto a Week 1."
    )
    parser.add_argument(
        "--filterindex0",
        default=None,
        help="Filtro opcional sobre la primera columna del fichero, Location/Orchard.",
    )
    return parser.parse_args(normalized_args)


def to_numeric_decimal_comma(frame: pd.DataFrame) -> pd.DataFrame:
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


def load_clean_and_make_deltas(
    filterindex0: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
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

    initial_rows = len(data)
    valid_label = data["week"].isin(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5+"])
    valid_sample = data["sample_number"].notna()
    valid_spectra = np.isfinite(data[feature_cols].to_numpy(dtype=float)).all(axis=1)
    positive_magnitude = (data[mag_cols].to_numpy(dtype=float) > 0).all(axis=1)
    base_keep = valid_label & valid_sample & valid_spectra & positive_magnitude

    removed_basic = data.loc[~base_keep, ["index0", "Harvest at ", "Sample No.", "week"]].copy()
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
        ["index0", "Harvest at ", "Sample No.", "week", "spectral_outlier_z"],
    ].copy()
    clean = clean.loc[clean["spectral_outlier_z"] <= OUTLIER_Z_THRESHOLD].copy()

    baseline = clean[clean["week"] == "Week 1"].copy()
    baseline = baseline.drop_duplicates(["index0", "sample_number"], keep="first")
    base_map = baseline.set_index(["index0", "sample_number"])[feature_cols]

    rows = []
    x_flat = []
    missing_baseline = []

    current = clean[clean["week"].isin(["Week 2", "Week 3", "Week 4", "Week 5+"])].copy()
    for idx, row in current.iterrows():
        key = (row["index0"], row["sample_number"])
        if key not in base_map.index:
            missing_baseline.append(row[["index0", "Harvest at ", "Sample No.", "week"]].to_dict())
            continue

        base_values = base_map.loc[key].to_numpy(dtype=float)
        current_values = row[feature_cols].to_numpy(dtype=float)

        base_mag = np.log10(base_values[: len(mag_cols)])
        current_mag = np.log10(current_values[: len(mag_cols)])
        base_phase = base_values[len(mag_cols) :]
        current_phase = current_values[len(mag_cols) :]

        delta_mag = current_mag - base_mag
        delta_phase = current_phase - base_phase
        rows.append(row)
        x_flat.append(np.hstack([current_mag, current_phase, delta_mag, delta_phase]))

    delta_data = pd.DataFrame(rows).reset_index(drop=True)
    x_flat_array = np.asarray(x_flat, dtype=float)

    print("=" * 80)
    print("LIMPIEZA Y DELTAS RESPECTO A WEEK 1")
    print("=" * 80)
    print(f"Filas iniciales con Sample No.: {initial_rows}")
    print(f"Eliminadas por etiqueta/muestra/espectro vacio/no finito/magnitud <= 0: {len(removed_basic)}")
    if len(removed_basic):
        print(removed_basic.head(60).to_string(index=False))
        if len(removed_basic) > 60:
            print(f"... {len(removed_basic) - 60} filas mas omitidas en consola")
    print(f"Eliminadas como outlier espectral robusto (z > {OUTLIER_Z_THRESHOLD}): {len(removed_outliers)}")
    if len(removed_outliers):
        print(removed_outliers.sort_values("spectral_outlier_z", ascending=False).to_string(index=False))
    print(f"Filas candidatas Week 2/3/4/5+ tras limpieza: {len(current)}")
    print(f"Eliminadas por no existir Week 1 completo del mismo index0 + sample: {len(missing_baseline)}")
    if missing_baseline:
        print(pd.DataFrame(missing_baseline).head(60).to_string(index=False))
        if len(missing_baseline) > 60:
            print(f"... {len(missing_baseline) - 60} filas mas omitidas en consola")
    print(f"Filas finales para modelado delta: {len(delta_data)}")
    print("Distribucion final por clase:")
    print(delta_data["week"].value_counts().sort_index().to_string())
    print()

    if len(delta_data) == 0:
        raise ValueError("No quedan filas para modelar despues de crear deltas respecto a Week 1.")

    return delta_data, x_flat_array, mag_cols, phase_cols


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Matriz de confusion (filas=real, columnas=predicho)")
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    print(cm.to_string())
    print()
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision macro: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall macro:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 macro:        {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print()


def save_confusion_matrices(records: list[dict[str, object]], output_path: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Archivo guardado: {output_path}")


def output_suffix(filterindex0: str | None) -> str:
    if not filterindex0:
        return "all"
    return re.sub(r"[^a-z0-9]+", "_", filterindex0.lower()).strip("_")


def main() -> None:
    args = parse_args()
    clean, x_flat, _, _ = load_clean_and_make_deltas(filterindex0=args.filterindex0)

    labels_order = ["Week 2", "Week 3", "Week 4", "Week 5+"]
    y_text = clean["week"].to_numpy()
    groups = (clean["index0"].astype(str) + "::" + clean["sample_number"].astype(str)).to_numpy()

    encoder = LabelEncoder()
    encoder.fit(labels_order)
    y = encoder.transform(y_text)
    n_splits = min(5, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    print(f"StratifiedGroupKFold con {n_splits} folds, shuffle=True, agrupado por index0 + Sample No.")
    print("Entrada: [impedancia_actual, impedancia_actual - impedancia_Week1] del mismo index0 + Sample No.")
    print("Clases: Week 2, Week 3, Week 4, Week 5+.")
    print("Normalizacion: StandardScaler en KNN/SVM/XGBoost; LabelEncoder para salida.")
    print()

    models = {
        "KNN_DELTA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "SVM_DELTA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced")),
            ]
        ),
        "XGBOOST_DELTA": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    confusion_records = []
    for name, model in models.items():
        pred = cross_val_predict(model, x_flat, y, groups=groups, cv=cv)
        y_true_text = encoder.inverse_transform(y)
        y_pred_text = encoder.inverse_transform(pred)
        print_metrics(name, y_true_text, y_pred_text, labels_order)
        cm = confusion_matrix(y_true_text, y_pred_text, labels=labels_order)
        for true_label, row in zip(labels_order, cm):
            for pred_label, count in zip(labels_order, row):
                confusion_records.append(
                    {"model": name, "true_label": true_label, "pred_label": pred_label, "count": int(count)}
                )

    save_confusion_matrices(
        confusion_records,
        OUTPUT_DIR / f"ripening_week_delta_confusion_matrices_{output_suffix(args.filterindex0)}.csv",
    )


if __name__ == "__main__":
    main()
