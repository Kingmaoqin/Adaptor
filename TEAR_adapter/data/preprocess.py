from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessorState:
    numeric_columns: List[str]
    categorical_columns: List[str]
    means: Dict[str, float]
    standard_deviations: Dict[str, float]
    categories: Dict[str, List[str]]


class TabularPreprocessor:
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.state: Optional[TabularPreprocessorState] = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
    ) -> None:
        means: Dict[str, float] = {}
        standard_deviations: Dict[str, float] = {}
        categories: Dict[str, List[str]] = {}
        for column_name in numeric_columns:
            series = pd.to_numeric(dataframe[column_name], errors="coerce")
            means[column_name] = float(series.mean()) if series.notna().any() else 0.0
            standard_deviations[column_name] = max(float(series.std()), 1.0) if series.notna().any() else 1.0
        for column_name in categorical_columns:
            series = dataframe[column_name].fillna("missing").astype(str)
            categories[column_name] = sorted(series.unique().tolist())
        self.state = TabularPreprocessorState(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            means=means,
            standard_deviations=standard_deviations,
            categories=categories,
        )

    def transform(self, dataframe: pd.DataFrame, columns: List[str]) -> np.ndarray:
        if self.state is None:
            raise ValueError("The preprocessor must be fitted before transform.")
        numeric_blocks = []
        categorical_blocks = []
        for column_name in columns:
            if column_name in self.state.numeric_columns:
                series = pd.to_numeric(dataframe[column_name], errors="coerce").fillna(self.state.means[column_name]).to_numpy()
                if self.standardize:
                    series = (series - self.state.means[column_name]) / self.state.standard_deviations[column_name]
                numeric_blocks.append(series.reshape(-1, 1))
            elif column_name in self.state.categorical_columns:
                vocab = self.state.categories[column_name]
                index_lookup = {value: index for index, value in enumerate(vocab)}
                series = dataframe[column_name].fillna("missing").astype(str)
                block = np.zeros((len(series), len(vocab)), dtype=np.float32)
                for row_index, value in enumerate(series):
                    block[row_index, index_lookup.get(value, 0)] = 1.0
                categorical_blocks.append(block)
        if numeric_blocks or categorical_blocks:
            return np.concatenate(numeric_blocks + categorical_blocks, axis=1).astype(np.float32)
        return np.zeros((len(dataframe), 0), dtype=np.float32)


def infer_column_types(
    dataframe: pd.DataFrame,
    excluded_columns: Iterable[str],
) -> Tuple[List[str], List[str]]:
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    for column_name in dataframe.columns:
        if column_name in excluded_columns:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column_name]):
            numeric_columns.append(column_name)
        else:
            categorical_columns.append(column_name)
    return numeric_columns, categorical_columns

