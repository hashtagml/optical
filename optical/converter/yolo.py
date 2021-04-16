"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""


import os
import imagesize
from pathlib import Path
import yaml
import warnings
from typing import Union
import numpy as np
import pandas as pd
from .base import FormatSpec
from .utils import exists, get_image_dir, get_annotation_dir


class Yolo(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self.class_file = Path(os.path.join(self.root, "data.yaml"))
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing 'images' directory."
        assert exists(self._annotation_dir), "root is missing 'annotations' directory."
        self._splits = self._find_splits()
        self._resolve_dataframe()

    def _find_splits(self):
        im_splits = [x.name for x in Path(self._image_dir).iterdir() if x.is_dir()]
        ann_splits = [x.name for x in Path(self._annotation_dir).iterdir() if x.is_dir()]

        if im_splits:
            self._has_image_split = True

        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {','.join(list(no_anns))}")
        return ann_splits

    def _resolve_dataframe(self):
        img_names = []
        cls_ids = []
        x_center = []
        y_center = []
        box_width = []
        box_height = []
        splits = []
        image_height = []
        image_width = []
        names_category = []
        for split in self._splits:
            ann_dir_files = os.path.join(self._annotation_dir, split)
            img_dir_files = os.path.join(self._image_dir, split)
            txt_files = [x for x in Path(ann_dir_files).glob("*.txt")]
            img_files = [x for x in Path(img_dir_files).glob("*.jpg")]
            for txt, img in zip(txt_files, img_files):
                file_names = os.path.basename(img)
                image_widths, image_heights = imagesize.get(img)
                image_height.append(image_heights)
                image_width.append(image_widths)
                with open(txt, 'rt') as fd:
                    first_line = fd.readline()
                    splited = first_line.split()
                    class_id = splited[0]
                    x_cent = splited[1]
                    y_cent = splited[2]
                    box_widths = splited[3]
                    box_heights = splited[4]
                    img_names.append(file_names)
                    x_center.append(x_cent)
                    y_center.append(y_cent)
                    box_width.append(box_widths)
                    box_height.append(box_heights)
                    cls_ids.append(class_id)
                    splits.append(split)

        if self.class_file.is_file():
            with open(self.class_file) as file:
                docs = yaml.load(file, Loader=yaml.FullLoader)
                class_names = docs["names"]
                for cls in cls_ids:
                    cat = class_names[int(cls)]
                    names_category.append(cat)
        if not self.class_file.is_file():
            category = [str(i) for i in cls_ids]
        else:
            category = [c for c in names_category]

        master_df = pd.DataFrame(
            list(
                zip(
                    img_names,
                    image_width,
                    image_height,
                    cls_ids,
                    category,
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                    splits,
                )
            ),
            columns=[
                 "image_id",
                 "image_width",
                 "image_height",
                 "class_id",
                 "category",
                 "x_min",
                 "y_min",
                 "width",
                 "height",
                 "split",
            ],
        )
        if len(master_df[pd.isnull(master_df.image_id)]) > 0:
            warnings.warn(
                    "There are annotations in your dataset for which there is no matching images"
                    + f"(in split '{split}'). These annotations will be removed during any "
                    + "computation or conversion. It is recommended that you clean your dataset."
            )
        for column in ["x_min", "y_min", "width", "height"]:
            master_df[column] = master_df[column].astype(np.float32)
        for column in ["image_width", "image_height"]:
            master_df[column] = master_df[column].astype(np.int32)
        for column in ["category"]:
            master_df[column] = master_df[column].astype(str)
        for column in ["class_id"]:
            master_df[column] = master_df[column].astype(np.int32)

        self.master_df = master_df
