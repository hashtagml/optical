"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import os
import random
from pathlib import Path
from typing import Optional, Union
import warnings

from numpy import imag

from ..visualizer.visualizer import Visualizer
from .coco import Coco
from .csv import Csv
from .utils import get_image_dir
from .yolo import Yolo

SUPPORTED_FORMATS = {"coco": Coco, "csv": Csv, "yolo": Yolo}


class Annotation:
    def __init__(self, root: str, format: str):
        if format.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"`{format}` is not a supported format")

        self.root = root
        self.format = format
        self.formatspec = SUPPORTED_FORMATS[format.lower()](root)

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

    def convert(self, to: str, output_dir: Optional[Union[str, os.PathLike]] = None, **kwargs):
        if not to.lower() in SUPPORTED_FORMATS:
            raise ValueError(f"`{format}` is not a supported conversion format")

        # if to.lower() == self.format.lower():
        #     print("Nice Try!")
        #     return

        # self.formatspec.convert(to.lower(), output_dir=output_dir, **kwargs)
        return self.formatspec.convert(to.lower(), output_dir=output_dir, **kwargs)

    def visualizer(
        self,
        images_dir: Optional[Union[str, os.PathLike]] = None,
        split: Optional[str] = None,
        img_size: Optional[int] = 512,
    ):
        if images_dir is None:
            if self.formatspec._has_image_split and split is not None:
                images_dir = get_image_dir(self.root) / split
            elif not self.formatspec._has_image_split:
                images_dir = get_image_dir(self.root)
            elif split is None:
                split = random.choice(list(self.formatspec.master_df.split.unique()))
                images_dir = get_image_dir(self.root) / split
                warnings.warn(
                    f"Since there is split given explicitly, {split} has been selected randomly. \
                    Please pass split argument if you want to visualize different split."
                )
        images_dir = Path(images_dir)
        return Visualizer(images_dir, self.formatspec.master_df, split, img_size)
