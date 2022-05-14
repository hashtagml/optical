"""
__author__: HashTagML
Created: Tuesday, 10th May 2022 12:16:28 am
"""

import random
import warnings
from pathlib import Path
from typing import Optional, Union

from ...visualization import Visualizer
from .base import FormatSpec
from .coco import Coco
from .createml import CreateML
from .csv import Csv
from .json_format import Json
from .pascal import Pascal
from .sagemaker import SageMaker
from .utils import CopyType, DetectionFormat, Pathlike, get_image_dir
from .yolo import Yolo

_TF_INSTALLED = True
try:
    import tensorflow  # noqa:F401

    from .tfrecord import Tfrecord
except ImportError:
    _TF_INSTALLED = False


def get_formatspec(format: DetectionFormat, root: Union[str, Path]) -> FormatSpec:
    if format == DetectionFormat.COCO:
        return Coco(root)
    if format == DetectionFormat.CREATEML:
        return CreateML(root)
    if format == DetectionFormat.CSV:
        return Csv(root)
    if format == DetectionFormat.PASCAL_VOC:
        return Pascal(root)
    if format == DetectionFormat.JSON:
        return Json(root)
    if format == DetectionFormat.SAGEMAKER_MANIFEST:
        return SageMaker(root)
    if format == DetectionFormat.YOLO:
        return Yolo(root)
    if format == DetectionFormat.TFRECORD:
        return Tfrecord(root)


class Annotation:
    def __init__(self, root: Pathlike, format: DetectionFormat):

        format = DetectionFormat(format)

        if format == DetectionFormat.TFRECORD and not _TF_INSTALLED:
            raise ImportError("Please Install Tensorflow for tfrecord support")
        self.root = root
        self.format = format
        self.formatspec = get_formatspec(format, root)

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

    def export(
        self,
        to: DetectionFormat,
        output_dir: Optional[Pathlike] = None,
        copy_images: Optional[CopyType] = None,
        prefix: Optional[str] = None,
    ):
        to = DetectionFormat(to)
        return self.formatspec.convert(
            to.lower(), root=self.root, output_dir=output_dir, prefix=prefix, copy_images=copy_images
        )

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
        image_dir: Optional[Pathlike] = None,
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
