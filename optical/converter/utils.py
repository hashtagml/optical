"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd


def ifnone(x: Any, y: Any, transform: Optional[Callable] = None, type_safe: bool = False):
    """if x is None return y otherwise x after applying transofrmation ``transform`` and
    casting the result back to original type if ``type_safe``

    Args:
        x (Any): returns x if x is not none
        y (Any): returns y if x is none
        transform (Optional[Callable], optional): applies transform to the output. Defaults to None.
        type_safe (bool, optional): if true, tries casting the output to the original type. Defaults to False.
    """

    if transform is not None:
        assert callable(transform), "`transform` should be either `None` or instance of `Callable`"
    else:

        def transform(x):
            return x

    if x is None:
        orig_type = type(y)
        out = transform(y)
    else:
        orig_type = type(x)
        out = transform(x)
    if type_safe:
        try:
            out = orig_type(out)
        except (ValueError, TypeError):
            warnings.warn(f"output could not be casted as type {orig_type.__name__}")
            pass
    return out


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


def write_coco_json(coco_dict: Dict, filename: Union[str, os.PathLike]):
    with open(filename, "w") as f:
        json.dump(coco_dict, f, indent=2)


def copyfile(
    src: Union[str, os.PathLike], dest: Union[str, os.PathLike], filename: Optional[Union[str, os.PathLike]] = None
) -> None:
    if filename is not None:
        filename = Path(src) / filename

    dest = Path(dest) / filename.name
    try:
        shutil.copyfile(filename, dest)
    except FileNotFoundError:
        pass
