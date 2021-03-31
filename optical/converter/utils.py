"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

from typing import Optional, Union
import os
import json
from pathlib import Path
import pandas as pd


def get_image_dir(root: Union[str, os.PathLike]):
    return Path(root) / "images"


def get_annotation_dir(root: Union[str, os.PathLike]):
    return Path(root) / "annotations"


def exists(path: Union[str, os.PathLike]):
    if Path(path).is_dir():
        return "dir"

    if Path(path).is_file():
        return "file"

    return


def read_coco(coco_json: Union[str, os.PathLike]):
    with open(coco_json, "r") as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"], coco["categories"]


def filter_split_category(df: pd.DataFrame, split: Optional[str] = None, category: Optional[str] = None):
    if split is not None:
        df = df.query("split == @split")

    if category is not None:
        if category not in df.category.unique():
            raise ValueError(f"class `{category}` is not present in annotations")
        df = df.query("category == @category")

    return df
