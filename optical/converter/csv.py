"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import os
from pathlib import Path
from typing import Union

import imagesize
import pandas as pd
import numpy as np

from ..visualizer.utils import check_df_cols
from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir


class Csv(FormatSpec):
    """Represents a CSV annotation object.

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
                    ├── train.csv
                    ├── valid.csv
                    └── test.csv

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
                    └── label.csv
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

    def _resolve_dataframe(self):
        columns = [
            "image_id",
            "image_path",
            "image_width",
            "image_height",
            "x_min",
            "y_min",
            "width",
            "height",
            "category",
            "class_id",
            "split",
        ]
        master_df = pd.DataFrame(columns=columns)
        req_cols = ["image_id", "x_min", "y_min", "x_max", "y_max", "category"]
        class_map = {}

        for split in self._splits:
            split_csv = self._annotation_dir / f"{split}.csv"
            split_df = pd.read_csv(split_csv)
            split_df_columns = split_df.columns.to_list()
            assert check_df_cols(
                split_df_columns, req_cols=req_cols
            ), f"Some required columns are not present in the {split_csv}.\
            Columns required for loading the annotations are {','.join(req_cols)}."
            im_paths = []
            if "image_path" not in split_df_columns:
                split_str = split if self._has_image_split else ""
                img_dir = Path(self._image_dir).joinpath(split_str)
                im_paths = split_df["image_id"].apply(lambda x: list(img_dir.glob(f"{x}"))[0])
                split_df["image_path"] = im_paths
            else:
                im_paths = list(split_df["image_path"].values)
            if "image_width" not in split_df_columns or "image_height" not in split_df_columns:
                im_dims = [imagesize.get(im_path) for im_path in im_paths]
                im_widths = [width for width, _ in im_dims]
                im_heights = [height for _, height in im_dims]
                split_df["image_width"] = im_widths
                split_df["image_height"] = im_heights
            split_df["width"] = split_df["x_max"] - split_df["x_min"]
            split_df["height"] = split_df["y_max"] - split_df["y_min"]
            split_df.drop(["x_max", "y_max"], axis=1, inplace=True)
            if not len(class_map):
                categories = split_df["category"].unique()
                class_map = dict(zip(categories, range(len(categories))))
            if "class_id" not in split_df_columns:
                split_df["class_id"] = split_df["category"].map(class_map)
            split_df.insert(len(split_df.columns.to_list()), "split", split)
            master_df = pd.concat([master_df, split_df], ignore_index=True)

        master_df = master_df[pd.notnull(master_df.image_id)]
        for col in ["x_min", "y_min", "width", "height"]:
            master_df[col] = master_df[col].astype(np.float32)

        for col in ["image_width", "image_height", "class_id"]:
            master_df[col] = master_df[col].astype(np.int32)
        self.master_df = master_df
