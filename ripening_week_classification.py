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

    parser = argparse.ArgumentParser(description="Clasificacion de semana de maduracion usando solo impedancia.")
    parser.add_argument(
        "--filterindex0",
        default=None,
        help="Filtro opcional sobre la primera columna del fichero, Location/Orchard.",
    )
    return parser.parse_args(normalized_args)


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    mad[mad == 0] = np.nan
    z = 0.6745 * (values - med) / mad
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def to_numeric_decimal_comma(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce"))


def load_and_clean(filterindex0: str | None = None) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    if TSV_FILE.exists():
        df = pd.read_csv(TSV_FILE, sep="\t")
        print(f"Leyendo datos desde TSV: {TSV_FILE}")
    elif XLSX_FILE.exists():
        df = pd.read_excel(XLSX_FILE, sheet_name="Sheet1")
        print(f"Leyendo datos desde Excel: {XLSX_FILE}")
    else:
        raise FileNotFoundError(f"No encuentro {TSV_FILE} ni {XLSX_FILE}")

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

    data["week"] = data["Harvest at "].map(normalize_week)
    data["sample_number"] = data["Sample No."].map(sample_number)
    data[feature_cols] = to_numeric_decimal_comma(data[feature_cols])

    initial_rows = len(data)
    valid_label = data["week"].isin(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5+"])
    valid_sample = data["sample_number"].notna()
    valid_spectra = np.isfinite(data[feature_cols].to_numpy(dtype=float)).all(axis=1)
    positive_magnitude = (data[mag_cols].to_numpy(dtype=float) > 0).all(axis=1)

    base_keep = valid_label & valid_sample & valid_spectra & positive_magnitude
    removed_basic = data.loc[~base_keep, ["Harvest at ", "Sample No.", "week"]].copy()
    clean = data.loc[base_keep].copy()

    mag = np.log10(clean[mag_cols].to_numpy(dtype=float))
    phase = clean[phase_cols].to_numpy(dtype=float)

    # Compact robust acquisition check: extreme level/slope/shape summaries often catch bad EIS reads.
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
    max_abs_z = np.max(np.abs(robust_z(summary)), axis=1)
    clean["spectral_outlier_z"] = max_abs_z
    is_outlier = clean["spectral_outlier_z"] > OUTLIER_Z_THRESHOLD
    removed_outliers = clean.loc[is_outlier, ["Harvest at ", "Sample No.", "week", "spectral_outlier_z"]].copy()
    clean = clean.loc[~is_outlier].copy()

    mag = np.log10(clean[mag_cols].to_numpy(dtype=float))
    phase = clean[phase_cols].to_numpy(dtype=float)
    x_flat = np.hstack([mag, phase])

    print("=" * 80)
    print("LIMPIEZA")
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
    print(f"Filas finales para modelado: {len(clean)}")
    print("Distribucion final por clase:")
    print(clean["week"].value_counts().sort_index().to_string())
    print()

    return clean, x_flat, mag_cols, phase_cols


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
    clean, x_flat, _, _ = load_and_clean(filterindex0=args.filterindex0)

    labels_order = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5+"]
    y_text = clean["week"].to_numpy()
    groups = clean["sample_number"].to_numpy()

    encoder = LabelEncoder()
    encoder.fit(labels_order)
    y = encoder.transform(y_text)
    n_splits = min(5, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    print(f"StratifiedGroupKFold con {n_splits} folds, shuffle=True, agrupado por Sample No.")
    print("Normalizacion: StandardScaler en KNN/SVM/XGBoost; LabelEncoder para salida.")
    print()

    models = {
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced")),
            ]
        ),
        "XGBOOST": Pipeline(
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
        OUTPUT_DIR / f"ripening_week_classification_confusion_matrices_{output_suffix(args.filterindex0)}.csv",
    )


if __name__ == "__main__":
    main()
