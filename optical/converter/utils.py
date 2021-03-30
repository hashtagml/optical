"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

from typing import Union
import os
import json
from pathlib import Path


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