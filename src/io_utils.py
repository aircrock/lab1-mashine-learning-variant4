from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from .config import output_path


def _auto_column_names(n: int) -> list[str]:
    return [f"col_{i + 1}" for i in range(n)]


def read_table(path: str | Path, has_header: bool = True) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    header = 0 if has_header else None

    if suffix == ".csv":
        df = pd.read_csv(path, header=header)
    elif suffix in {".txt", ".tsv"}:
        df = pd.read_csv(path, sep=None, engine="python", header=header)
    elif suffix == ".xlsx":
        df = pd.read_excel(path, header=header, engine="openpyxl")
    elif suffix == ".mat":
        mat = loadmat(path, squeeze_me=True)
        arr = np.asarray(mat.get("table"))
        if arr.ndim != 2:
            raise ValueError("В .mat не найден двумерный массив table")
        raw_cols = mat.get("columns")
        if raw_cols is None:
            cols = _auto_column_names(arr.shape[1])
        else:
            cols = [str(c).strip() for c in np.atleast_1d(raw_cols).tolist()]
            if len(cols) != arr.shape[1]:
                cols = _auto_column_names(arr.shape[1])
        df = pd.DataFrame(arr, columns=cols)
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    if not has_header:
        df.columns = _auto_column_names(df.shape[1])
    return df


def write_main_outputs(
    df: pd.DataFrame,
    scaling_comparison: pd.DataFrame | None = None,
    inverse_comparison: pd.DataFrame | None = None,
    base_name: str = "variant4",
) -> dict[str, str]:
    """
    Создаёт РОВНО четыре табличных файла задания 2.
    Дополнительные результаты задания 34/35 кладутся листами в тот же XLSX,
    чтобы не создавать лишних файлов в output/.
    """
    paths: dict[str, str] = {}

    xlsx_path = output_path(base_name, "xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        if scaling_comparison is not None:
            scaling_comparison.to_excel(writer, sheet_name="task34_scaling", index=False)
        if inverse_comparison is not None:
            inverse_comparison.to_excel(writer, sheet_name="task35_inverse", index=False)
    paths["xlsx"] = str(xlsx_path)

    csv_path = output_path(base_name, "csv")
    df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)

    txt_path = output_path(base_name, "txt")
    df.to_csv(txt_path, index=False, sep="\t")
    paths["txt"] = str(txt_path)

    mat_path = output_path(base_name, "mat")
    numeric = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    savemat(mat_path, {"table": numeric, "columns": np.asarray(df.columns, dtype=object)})
    paths["mat"] = str(mat_path)
    return paths


def check_data_quality(df: pd.DataFrame, expected_schema: dict[str, str]) -> dict:
    report = {
        "shape": list(df.shape),
        "missing": df.isna().sum().astype(int).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "invalid_numeric": {},
        "invalid_datetime": {},
        "type_mismatches": {},
    }

    for col, expected in expected_schema.items():
        if col not in df.columns:
            report["type_mismatches"][col] = "столбец отсутствует"
            continue

        if expected == "numeric":
            converted = pd.to_numeric(df[col], errors="coerce")
            invalid = converted.isna() & df[col].notna()
            if invalid.any():
                report["invalid_numeric"][col] = df.loc[invalid, col].astype(str).tolist()
            if not pd.api.types.is_numeric_dtype(df[col]):
                report["type_mismatches"][col] = f"ожидался numeric, фактически {df[col].dtype}"
        elif expected == "integer":
            if not pd.api.types.is_integer_dtype(df[col]):
                report["type_mismatches"][col] = f"ожидался integer, фактически {df[col].dtype}"
        elif expected == "datetime":
            parsed = pd.to_datetime(df[col], errors="coerce")
            invalid = parsed.isna() & df[col].notna()
            if invalid.any():
                report["invalid_datetime"][col] = df.loc[invalid, col].astype(str).tolist()
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                report["type_mismatches"][col] = f"ожидался datetime, фактически {df[col].dtype}"
        elif expected == "category":
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                report["type_mismatches"][col] = f"ожидался category, фактически {df[col].dtype}"

    report["has_problems"] = (
        any(v > 0 for v in report["missing"].values())
        or bool(report["invalid_numeric"])
        or bool(report["invalid_datetime"])
        or bool(report["type_mismatches"])
    )
    if report["has_problems"]:
        warnings.warn("В датасете обнаружены пропуски/некорректные данные/несовпадения типов.")
    return report


def extract_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют столбцы варианта: {missing}")
    return df.loc[:, columns].copy()


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    print(df.head(n).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)
