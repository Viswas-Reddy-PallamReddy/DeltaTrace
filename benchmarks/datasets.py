"""Loaders for famous public datasets that have **no primary key**.

Each dataset is downloaded once from the seaborn-data mirror and cached locally.
The :class:`DatasetSpec` records the domain knowledge a record-matching system
needs anyway: which columns are stable enough to *block* candidate matches, and
which columns a realistic edit would touch.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

_MIRROR = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
_CACHE = Path(__file__).resolve().parent / "_cache"


@dataclass
class DatasetSpec:
    """How one keyless dataset is loaded and evolved in the benchmark."""

    name: str
    csv: str
    block_on: List[str]
    update_cols: List[str]
    why_keyless: str
    sample: Optional[int] = None          # cap rows (keeps notebooks snappy)
    drop_cols: List[str] = field(default_factory=list)
    url: Optional[str] = None             # full override (non-seaborn sources)
    header: object = "infer"              # pandas read_csv header arg
    names: Optional[List[str]] = None     # column names for headerless sources
    skipinitialspace: bool = False
    na_values: Optional[List[str]] = None


_ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


DATASETS = {
    "iris": DatasetSpec(
        name="iris",
        csv="iris.csv",
        block_on=["species"],
        update_cols=["sepal_length", "petal_length"],
        why_keyless="150 flower measurements; a row is just four numbers + a "
        "species label. Two flowers can share identical measurements, so nothing "
        "uniquely identifies a row.",
    ),
    "penguins": DatasetSpec(
        name="penguins",
        csv="penguins.csv",
        block_on=["species", "island"],
        update_cols=["body_mass_g", "bill_length_mm"],
        why_keyless="344 penguin observations with missing values and no tag/ID; "
        "identity is purely the (often repeated) measurements.",
    ),
    "titanic": DatasetSpec(
        name="titanic",
        csv="titanic.csv",
        block_on=["pclass", "sex"],
        update_cols=["age", "fare"],
        why_keyless="891 passengers with no manifest ID; many share class, sex, "
        "age and fare, and ages are frequently corrected.",
        drop_cols=["alive"],  # leakage-ish duplicate of 'survived'
    ),
    "diamonds": DatasetSpec(
        name="diamonds",
        csv="diamonds.csv",
        block_on=["cut", "color", "clarity"],
        update_cols=["price", "carat"],
        why_keyless="~54k diamonds described only by physical attributes; "
        "thousands of rows are exact duplicates and prices get re-graded.",
        sample=20000,
    ),
    "adult": DatasetSpec(
        name="adult",
        csv="adult.data",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        names=_ADULT_COLS,
        skipinitialspace=True,
        na_values=["?"],
        block_on=["workclass", "education", "sex"],
        update_cols=["hours_per_week", "age"],
        why_keyless="~32k US census respondents described only by demographics; "
        "there is no respondent ID and large numbers of people share an identical "
        "attribute profile.",
        drop_cols=["fnlwgt"],  # sampling weight, not a real attribute
        sample=20000,
    ),
}


def _download(spec: DatasetSpec) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / spec.csv
    if not dest.exists():
        url = spec.url or (_MIRROR + spec.csv)
        raw = urllib.request.urlopen(url, timeout=60).read()
        dest.write_bytes(raw)
    return dest


def load(name: str, *, seed: int = 0) -> pd.DataFrame:
    """Return a clean, keyless DataFrame for ``name`` (downloaded + cached)."""
    spec = DATASETS[name]
    df = pd.read_csv(
        io.BytesIO(_download(spec).read_bytes()),
        header=spec.header,
        names=spec.names,
        skipinitialspace=spec.skipinitialspace,
        na_values=spec.na_values,
    )

    # strip any unnamed index column some CSV mirrors carry
    junk = [c for c in df.columns if str(c).startswith("Unnamed") or c == ""]
    df = df.drop(columns=junk + [c for c in spec.drop_cols if c in df.columns])

    if spec.sample is not None and len(df) > spec.sample:
        df = df.sample(n=spec.sample, random_state=seed)

    df = df.reset_index(drop=True)
    return df
