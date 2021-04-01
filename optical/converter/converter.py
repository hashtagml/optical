"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import os
import warnings
from pathlib import Path, PosixPath
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .utils import ifnone


class LabelEncoder:
    def __init__(self):
        self._map = dict()

    def fit(self, series):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        categories = series.unique().tolist()
        label_map = dict(zip(categories, np.arange(len(categories))))
        for k, v in label_map.items():
            if k not in self._map:
                self._map[k] = label_map[k]

    def transform(self, series):
        series = series.map(self._map)
        return series

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)


def write_yolo_txt(filename: str, output_dir: Union[str, os.PathLike, PosixPath], yolo_string: str):
    filepath = Path(output_dir).joinpath(Path(filename).stem + ".txt")
    with open(filepath, "a") as f:
        f.write(yolo_string)
        f.write("\n")


def convert_yolo(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):

    output_dir = ifnone(output_dir, Path(root) / "yolo" / "annotations", Path)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = df.split.unique().tolist()
    lbl = LabelEncoder()

    for split in splits:
        output_subdir = output_dir / split
        output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()
        hw_missing = split_df[pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])]

        split_df["x_center"] = split_df["x_min"] + split_df["width"] / 2
        split_df["y_center"] = split_df["y_min"] + split_df["height"] / 2

        # normalize
        split_df["x_center"] = split_df["x_center"] / split_df["image_width"]
        split_df["y_center"] = split_df["y_center"] / split_df["image_height"]
        split_df["width"] = split_df["width"] / split_df["image_width"]
        split_df["height"] = split_df["height"] / split_df["image_height"]

        split_df["class_index"] = lbl.fit_transform(split_df["category"])

        split_df["yolo_string"] = (
            split_df["class_index"].astype(str)
            + " "
            + split_df["x_center"].astype(str)
            + " "
            + split_df["y_center"].astype(str)
            + " "
            + split_df["width"].astype(str)
            + " "
            + split_df["height"].astype(str)
        )

        ds = split_df.groupby("image_id")["yolo_string"].agg(lambda x: "\n".join(x)).reset_index()

        image_ids = ds["image_id"].tolist()
        yolo_strings = ds["yolo_string"].tolist()

        for image_id, ystr in tqdm(zip(image_ids, yolo_strings), total=len(image_ids), desc=f"split: {split}"):
            write_yolo_txt(image_id, output_subdir, ystr)


def convert_csv(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):

    df["x_max"] = df["x_min"] + df["width"]
    df["y_max"] = df["y_min"] + df["height"]
    df.drop(["width", "height"], axis=1, inplace=True)
    for col in ("x_min", "y_min", "x_max", "y_max"):
        df[col] = df[col].astype(np.int32)

    output_dir = ifnone(output_dir, Path(root) / "csv" / "annotations", Path)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split")
        split_df.drop(["split"], axis=1, inplace=True)
        split_df.to_csv(output_dir.joinpath(f"{split}.csv"), index=False)
