from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cast_metadata_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "id" in out:
        out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "category" in out:
        out["category"] = out["category"].astype("category")
    return out


def cast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def handle_problem(df: pd.DataFrame, col: str, action: str, value: float | None = None) -> pd.DataFrame:
    out = df.copy()
    action = action.lower()
    if action == "drop":
        return out.dropna(subset=[col]).reset_index(drop=True)
    if action == "cast":
        out[col] = pd.to_numeric(out[col], errors="coerce")
    elif action == "mean":
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].mean())
    elif action == "ffill":
        out[col] = out[col].ffill().bfill()
    elif action == "bfill":
        out[col] = out[col].bfill().ffill()
    elif action == "zero":
        out[col] = out[col].fillna(0)
    elif action == "value":
        if value is None:
            raise ValueError("Для action='value' нужно указать value")
        out[col] = out[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")
    return out


def interactive_cleaning_menu(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    out = cast_numeric_columns(df, numeric_cols)
    menu = {"1": "drop", "2": "cast", "3": "mean", "4": "ffill", "5": "bfill", "6": "zero", "7": "value"}
    for col in numeric_cols:
        while out[col].isna().any():
            print(f"\nСтолбец {col}: проблемных значений = {int(out[col].isna().sum())}")
            print("1 удалить строки | 2 привести тип | 3 среднее | 4 предыдущее | 5 последующее | 6 нули | 7 значение")
            choice = input("Выбор: ").strip()
            if choice not in menu:
                print("Некорректный пункт меню.")
                continue
            value = float(input("Введите значение: ")) if menu[choice] == "value" else None
            out = handle_problem(out, col, menu[choice], value)
            if menu[choice] == "cast" and out[col].isna().any():
                print("Приведение выполнено, но NaN остались — выберите заполнение или удаление.")
    return out


def automatic_cleaning(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    out = cast_numeric_columns(df, numeric_cols)
    actions = {}
    for col in numeric_cols:
        count = int(out[col].isna().sum())
        if count:
            mean_value = float(out[col].mean())
            out[col] = out[col].fillna(mean_value)
            actions[col] = {"action": "mean", "count": count, "value": mean_value}
    return out, actions


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64, copy=True)


@dataclass
class MinMaxScalerCustom:
    a: float = 0.0
    b: float = 1.0
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Сначала вызовите fit")
        span = self.max_ - self.min_
        safe_span = np.where(span == 0, 1.0, span)
        z = self.a + (np.asarray(x, dtype=np.float64) - self.min_) * (self.b - self.a) / safe_span
        if np.any(span == 0):
            z[..., span == 0] = self.a
        return z

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Сначала вызовите fit")
        if self.a == self.b:
            raise ValueError("a и b не должны совпадать")
        return self.min_ + (np.asarray(z, dtype=np.float64) - self.a) * (self.max_ - self.min_) / (self.b - self.a)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class StandardScalerCustom:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Сначала вызовите fit")
        return (np.asarray(x, dtype=np.float64) - self.mean_) / np.where(self.std_ == 0, 1.0, self.std_)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Сначала вызовите fit")
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def split_train_val_test(x: np.ndarray, ratios=(0.7, 0.15, 0.15)):
    r = np.asarray(ratios, dtype=np.float64)
    r = r / r.sum()
    n = len(x)
    n_train = int(np.floor(n * r[0]))
    n_val = int(np.floor(n * r[1]))
    return x[:n_train], x[n_train:n_train+n_val], x[n_train+n_val:]


def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Ожидается одномерный массив")
    return np.lib.stride_tricks.sliding_window_view(x, width)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    return sliding_window(x, width).mean(axis=1)
