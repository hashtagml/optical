"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import os
import warnings
from pathlib import Path
from typing import Union
import numpy as np

import pandas as pd
import xml.etree.ElementTree as ET

from .base import FormatSpec
from .utils import exists, get_image_dir, get_annotation_dir


class Pascal(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._splits = self._find_splits()
        self._resolve_dataframe()

    def _find_splits(self):
        im_splits = [x.name for x in Path(self._image_dir).iterdir() if x.is_dir()]
        ann_splits = [x.name for x in Path(self._annotation_dir).iterdir() if x.is_dir()]

        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {','.join(list(no_anns))}")
        return ann_splits

    def _resolve_dataframe(self):
        img_filenames = []
        img_widths = []
        img_heights = []
        cls_names = []
        x_mins = []
        y_mins = []
        box_widths = []
        box_heights = []
        splits = []
        for split in self._splits:
            folder_files = self._annotation_dir / f"{split}"
            xml_files = [x for x in Path(folder_files).glob("*.xml")]
            for xml in xml_files:
                tree = ET.parse(xml)
                root = tree.getroot()
                img_filename = root.find("filename").text
                img_width = root.find("size").find("width").text
                img_height = root.find("size").find("height").text
                for obj in root.findall("object"):
                    cls_name = obj.find("name").text
                    x_min = int(obj.find("bndbox").find("xmin").text)
                    y_min = int(obj.find("bndbox").find("ymin").text)
                    box_width = int(obj.find("bndbox").find("xmax").text) - int(x_min)
                    box_height = int(obj.find("bndbox").find("ymax").text) - int(y_min)
                    img_filenames.append(img_filename)
                    img_widths.append(img_width)
                    img_heights.append(img_height)
                    cls_names.append(cls_name)
                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    box_widths.append(box_width)
                    box_heights.append(box_height)
                    splits.append(split)
        class_dict = dict(zip(set(cls_names), [i for i in range(len(set(cls_names)))]))
        class_ids = [class_dict[cate] for cate in cls_names]
        master_df = pd.DataFrame(
            list(
                zip(
                    img_filenames,
                    img_widths,
                    img_heights,
                    x_mins,
                    y_mins,
                    box_widths,
                    box_heights,
                    cls_names,
                    class_ids,
                    splits,
                )
            ),
            columns=[
                "image_id",
                "image_width",
                "image_height",
                "x_min",
                "y_min",
                "width",
                "height",
                "category",
                "class_id",
                "split",
            ],
        )
        for col in ["x_min", "y_min", "width", "height"]:
            master_df[col] = master_df[col].astype(np.float32)
        for col in ["image_width", "image_height"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df
