"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""


import os
import warnings
from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir, read_coco


class Coco(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._splits = self._find_splits()
        self._resolve_dataframe()

    def _find_splits(self):
        im_splits = [x.name for x in Path(self._image_dir).iterdir() if x.is_dir()]
        ann_splits = [x.stem for x in Path(self._annotation_dir).glob("*.json")]

        if im_splits:
            self._has_image_split = True

        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {', '.join(list(no_anns))}")
        return ann_splits

    def _get_class_map(self, categories: List):
        class_map = dict()
        for cat in categories:
            class_map[cat["id"]] = cat["name"]
        return class_map

    def _resolve_dataframe(self):
        split_str = []
        master_df = pd.DataFrame(
            columns=["image_id", "image_width", "image_height", "x_min", "y_min", "width", "height", "category"],
        )
        for split in self._splits:

            coco_json = self._annotation_dir / f"{split}.json"
            images, annots, cats = read_coco(coco_json)
            split_str.append([split, len(images), len(annots), len(cats)])

            class_map = self._get_class_map(cats)

            images_df = pd.DataFrame(images)
            images_df = images_df[["id", "file_name", "width", "height"]]
            images_df.rename(columns={"width": "image_width", "height": "image_height"}, inplace=True)

            instances = [(x["image_id"], x["category_id"], x["bbox"]) for x in annots]
            annots_df = pd.DataFrame(instances, columns=["image_id", "class_id", "bbox"])
            annots_df["category"] = annots_df["class_id"].map(class_map)
            annots_df[["x_min", "y_min", "width", "height"]] = pd.DataFrame(
                annots_df["bbox"].to_list(), index=annots_df.index
            )

            annots_df.drop(["bbox"], axis=1, inplace=True)

            annots_df = annots_df.merge(images_df, left_on="image_id", right_on="id", how="left")
            annots_df.drop(["id", "image_id"], axis=1, inplace=True)
            annots_df.rename(columns={"file_name": "image_id"}, inplace=True)
            annots_df["split"] = split

            if len(annots_df[pd.isnull(annots_df.image_id)]) > 0:
                warnings.warn(
                    "There are annotations in your dataset for which there is no matching images"
                    + f"(in split `{split}`). These annotations will be removed during any "
                    + "computation or conversion. It is recommended that you clean your dataset."
                )

            master_df = pd.concat([master_df, annots_df], ignore_index=True)

        master_df = master_df[pd.notnull(master_df.image_id)]
        for col in ["x_min", "y_min", "width", "height"]:
            master_df[col] = master_df[col].astype(np.float32)

        for col in ["image_width", "image_height"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df
