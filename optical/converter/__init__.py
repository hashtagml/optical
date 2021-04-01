"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import os
from typing import Optional, Union
from .coco import Coco
from .yolo import Yolo
from .csv import Csv

SUPPORTED_FORMATS = {"coco": Coco, "csv": Csv, "yolo": Yolo}


class Annotation:
    def __init__(self, root: str, format: str):
        if format.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"`{format}` is not a supported format")

        self.root = root
        self.format = format
        self.formatspec = SUPPORTED_FORMATS[format.lower()](root)

    def describe(self):
        return self.formatspec.describe()

    def show_distribution(self):
        return self.formatspec.show_distribution()

    def bbox_scatter(self, split: Optional[str] = None, category: Optional[str] = None):
        return self.formatspec.bbox_scatter(split, category)

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None):
        return self.formatspec.bbox_stats(split, category)

    def convert(self, to: str, output_dir: Optional[Union[str, os.PathLike]] = None):
        if not to.lower() in SUPPORTED_FORMATS:
            raise ValueError(f"`{format}` is not a supported conversion format")

        if to.lower() == self.format.lower():
            print("Nice Try!")
            return

        self.formatspec.convert(to.lower(), output_dir)
