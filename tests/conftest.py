"""
__author__: HashTagML
license: MIT
Created: Thursday, 1st January 1970
"""

from pathlib import Path
from typing import Union
import pytest
import numpy as np
import pandas as pd
import os
import shutil

testdir = Path("~").expanduser().joinpath(".optical").joinpath("tests")


def touch(path: Union[str, os.PathLike]):
    Path(path).absolute().parent.mkdir(exist_ok=True, parents=True)
    Path(path).touch()
    return path


@pytest.fixture()
def dirs(format: str, has_split: bool):
    exts = {"coco": "json", "yolo": "txt", "pascal": "xml", "createml": "json", "sagemaker": "manifest", "csv": "csv"}

    format_indiv = ("pascal", "yolo")
    format_consolidated = ("coco", "sagemaker", "createml", "csv")

    if has_split:
        touch(Path(testdir).joinpath(format).joinpath("images").joinpath("train").joinpath("1.jpg"))
        touch(Path(testdir).joinpath(format).joinpath("images").joinpath("valid").joinpath("2.jpg"))
        touch(Path(testdir).joinpath(format).joinpath("images").joinpath("test").joinpath("3.jpg"))

        if format in format_consolidated:
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"train.{exts[format]}"))
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"valid.{exts[format]}"))
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"test.{exts[format]}"))

        elif format in format_indiv:
            touch(
                Path(testdir).joinpath(format).joinpath("annotations").joinpath("train").joinpath(f"1.{exts[format]}")
            )
            touch(
                Path(testdir).joinpath(format).joinpath("annotations").joinpath("valid").joinpath(f"2.{exts[format]}")
            )
            touch(
                Path(testdir).joinpath(format).joinpath("annotations").joinpath("test").joinpath(f"3.{exts[format]}")
            )

        splits = ["train", "valid", "test"]

    else:
        for i in range(3):
            touch(Path(testdir).joinpath(format).joinpath("images").joinpath(f"{i}.jpg"))

        if format in format_consolidated:
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"label.{exts[format]}"))
            splits = ["label"]
        elif format in format_indiv:
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"1.{exts[format]}"))
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"2.{exts[format]}"))
            touch(Path(testdir).joinpath(format).joinpath("annotations").joinpath(f"3.{exts[format]}"))
            splits = ["main"]

    yield Path(testdir).joinpath(format), format, splits, has_split
    shutil.rmtree(Path(testdir).joinpath(format))


@pytest.fixture()
def make_df():
    image_ids = [f"{i.jpg}" for i in range(30)]
    image_width = image_height = [1024] * 30
    x_min = np.random.randint(low=0, high=1000, size=30)
    y_min = np.random.randint(low=0, high=1000, size=30)
    widths = np.random.randint(low=20, high=200, size=30)
    heights = np.random.randint(low=20, high=200, size=30)
    class_ids = np.random.randint(low=1, high=5, size=30)
    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "image_width": image_width,
            "image_height": image_height,
            "x_min": x_min,
            "y_min": y_min,
            "width": widths,
            "height": heights,
            "class_id": class_ids,
        }
    )
    df["category"] = df["class_id"].astype(str)
    df["split"] = np.random.choice(["train", "valid", "test"], size=30)
    df["image_path"] = ""
    return df
