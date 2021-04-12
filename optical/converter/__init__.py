"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import os
from typing import Optional, Union

from .coco import Coco
from .csv import Csv
from .yolo import Yolo
from ..visualizer.visualizer import Visualizer
from .utils import get_image_dir, ifnone

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
        images_dir = ifnone(images_dir, get_image_dir(self.root))
        return Visualizer(images_dir, self.formatspec.master_df, split, img_size)
