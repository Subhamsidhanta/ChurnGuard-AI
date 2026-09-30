"""Preprocessing pipeline construction."""
from typing import Iterable
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(categorical: Iterable[str], numeric: Iterable[str]) -> ColumnTransformer:
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    return ColumnTransformer([("categorical", cat, list(categorical)),
                              ("numeric", num, list(numeric))])
