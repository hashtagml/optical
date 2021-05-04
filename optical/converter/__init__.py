"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""
# TODO: needs better solution for Handling TFrecords

import os
import random
from pathlib import Path
from typing import Optional, Union
import warnings

from .coco import Coco
from .csv import Csv
from .yolo import Yolo
from .pascal import Pascal
from .sagemaker import SageMaker
from .createml import CreateML
from .tfrecord import Tfrecord
from .simple_json import SimpleJson
from ..visualizer.visualizer import Visualizer
from .utils import get_image_dir, ifnone

from ..converter.base import FormatSpec

_TF_INSTALLED = True
try:
    import tensorflow as tf
except ImportError:
    _TF_INSTALLED = False

SUPPORTED_FORMATS = {
    "coco": Coco,
    "csv": Csv,
    "yolo": Yolo,
    "sagemaker": SageMaker,
    "pascal": Pascal,
    "tfrecord": Tfrecord,
    "createml": CreateML,
    "simple_json": SimpleJson,
}


class Annotation:
    def __init__(self, root: str, format: str):
        if format.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"`{format}` is not a supported format")

        if format.lower() == "tfrecord" and not _TF_INSTALLED:
            raise ImportError("Please Install Tensorflow for tfrecord support")
        self.root = root
        self.format = format
        self.formatspec = SUPPORTED_FORMATS[format.lower()](root)

    def __str__(self):
        return self.formatspec.__str__()

    def __repr__(self):
        return self.formatspec.__repr__()

    @property
    def splits(self):
        return self.formatspec.splits

    @property
    def label_df(self):
        return self.formatspec.master_df

    def describe(self):
        return self.formatspec.describe()

    def show_distribution(self):
        return self.formatspec.show_distribution()

    def bbox_scatter(self, split: Optional[str] = None, category: Optional[str] = None):
        return self.formatspec.bbox_scatter(split, category)

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None):
        return self.formatspec.bbox_stats(split, category)

    def export(self, to: str, output_dir: Optional[Union[str, os.PathLike]] = None, **kwargs):
        if not to.lower() in SUPPORTED_FORMATS:
            raise ValueError(f"`{to}` is not a supported conversion format")

        return self.formatspec.convert(to.lower(), output_dir=output_dir, **kwargs)

    def train_test_split(self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42):
        """splits the dataset into train and validation sets

        Args:
            test_size (float, optional): Fraction of total images to be kept for validation. Defaults to 0.2.
            stratified (bool, optional): Whether to stratify the split. Defaults to False.
            random_state (int, optional): random state for the split. Defaults to 42.

        Returns:
            FormatSpec: Returns an instance of `FormatSpec` class
        """
        return self.formatspec.split(test_size, stratified, random_state)

    def visualizer(
        self,
        image_dir: Optional[Union[str, os.PathLike]] = None,
        split: Optional[str] = None,
        img_size: Optional[int] = 512,
        **kwargs,
    ):
        if image_dir is None:
            random_split = random.choice(list(self.formatspec.master_df.split.unique()))
            if split is None:
                split = random_split
                warnings.warn(
                    f"Since there is not split specified explicitly, {split} has been selected randomly."
                    + "Please specify split if you want to visualize different split."
                )
            if self.formatspec._has_image_split:
                image_dir = get_image_dir(self.root) / split
            else:
                image_dir = get_image_dir(self.root)
        image_dir = Path(image_dir)
        return Visualizer(image_dir, self.formatspec.master_df, split, img_size, **kwargs)
