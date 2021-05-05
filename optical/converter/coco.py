"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""


import os
import warnings

# from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir, read_coco


class Coco(FormatSpec):
    """Represents a COCO annotation object.

    Args:
        root (Union[str, os.PathLike]): path to root directory. Expects the ``root`` directory to have either
           of the following layouts:

           .. code-block:: bash

                root
                ├── images
                │   ├── train
                │   │   ├── 1.jpg
                │   │   ├── 2.jpg
                │   │   │   ...
                │   │   └── n.jpg
                │   ├── valid (...)
                │   └── test (...)
                │
                └── annotations
                    ├── train.json
                    ├── valid.json
                    └── test.json

            or,

            .. code-block:: bash

                root
                ├── images
                │   ├── 1.jpg
                │   ├── 2.jpg
                │   │   ...
                │   └── n.jpg
                │
                └── annotations
                    └── label.json
    """

    def __init__(self, root: Union[str, os.PathLike]):
        # self.root = Path(root)
        super().__init__(root)
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._find_splits()
        self._resolve_dataframe()

    def _get_class_map(self, categories: List):
        """map from category id to category name"""
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
            split_dir = split if self._has_image_split else ""
            annots_df["image_path"] = annots_df["image_id"].map(
                lambda x: self.root.joinpath("images").joinpath(split_dir).joinpath(x)
            )

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

        for col in ["image_width", "image_height", "class_id"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df
