import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from kaggle.api.kaggle_api_extended import KaggleApi
from optical.detection.formats.utils import (DetectionFormat, Pathlike,
                                             file_extensions)

testdir = Path("~").expanduser().joinpath(".optical").joinpath("tests")


@pytest.fixture()
def test_dir():
    return testdir


def touch(path: Pathlike):
    Path(path).absolute().parent.mkdir(exist_ok=True, parents=True)
    Path(path).touch()
    return path


@pytest.fixture()
def dirs(format: DetectionFormat, has_split: bool):

    format_indiv = (DetectionFormat.PASCAL_VOC, DetectionFormat.YOLO)
    format_consolidated = (
        DetectionFormat.COCO,
        DetectionFormat.CSV,
        DetectionFormat.CREATEML,
        DetectionFormat.SAGEMAKER_MANIFEST,
    )

    if has_split:
        touch(Path(testdir).joinpath(format.value).joinpath("images").joinpath("train").joinpath("1.jpg"))
        touch(Path(testdir).joinpath(format.value).joinpath("images").joinpath("valid").joinpath("2.jpg"))
        touch(Path(testdir).joinpath(format.value).joinpath("images").joinpath("test").joinpath("3.jpg"))

        if format in format_consolidated:
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath(f"train.{file_extensions[format]}")
            )
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath(f"valid.{file_extensions[format]}")
            )
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath(f"test.{file_extensions[format]}")
            )

        elif format in format_indiv:
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath("train")
                .joinpath(f"1.{file_extensions[format]}")
            )
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath("valid")
                .joinpath(f"2.{file_extensions[format]}")
            )
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath("test")
                .joinpath(f"3.{file_extensions[format]}")
            )

        splits = ["train", "valid", "test"]

    else:
        for i in range(3):
            touch(Path(testdir).joinpath(format.value).joinpath("images").joinpath(f"{i}.jpg"))

        if format in format_consolidated:
            touch(
                Path(testdir)
                .joinpath(format.value)
                .joinpath("annotations")
                .joinpath(f"label.{file_extensions[format]}")
            )
            splits = ["label"]
        elif format in format_indiv:
            touch(
                Path(testdir).joinpath(format.value).joinpath("annotations").joinpath(f"1.{file_extensions[format]}")
            )
            touch(
                Path(testdir).joinpath(format.value).joinpath("annotations").joinpath(f"2.{file_extensions[format]}")
            )
            touch(
                Path(testdir).joinpath(format.value).joinpath("annotations").joinpath(f"3.{file_extensions[format]}")
            )
            splits = ["main"]

    yield Path(testdir).joinpath(format.value), format, splits, has_split
    shutil.rmtree(Path(testdir).joinpath(format.value))


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


@pytest.fixture()
def root(fmt: str, has_split: bool):
    if not Path(testdir).joinpath("testfiles").is_dir():
        get_test_files()

    root_ = Path(testdir) / "testfiles"
    src_im = "has_split" if has_split else "flat"
    dest_dir = "splits" if has_split else "flats"
    dest = root_.joinpath(dest_dir).joinpath(fmt).joinpath("images")
    src = Path(testdir).joinpath("testfiles").joinpath("images").joinpath(src_im)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return root_.joinpath(dest_dir).joinpath(fmt)


def get_test_files():
    print("Downloading testfiles...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("bishwarup/objdet-formats", path=str(testdir), unzip=True)


@pytest.fixture()
def label_df():
    if not Path(testdir).joinpath("testfiles").is_dir():
        get_test_files()

    df = pd.read_csv(Path(testdir).joinpath("testfiles").joinpath("label_df.csv"))
    return df
