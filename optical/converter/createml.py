"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import json
import os
import warnings
from typing import Union

import imagesize
import pandas as pd

from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir


class CreateML(FormatSpec):
    """Class to handle createML json annotation transformations

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
        # self.root = root
        super().__init__(root)
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._find_splits()
        self._resolve_dataframe()

    def _resolve_dataframe(self):
        master_data = {
            "image_id": [],
            "image_path": [],
            "image_width": [],
            "image_height": [],
            "x_min": [],
            "y_min": [],
            "width": [],
            "height": [],
            "category": [],
            "split": [],
        }

        # checking if there is splitting or not

        for split in self._splits:
            image_dir = self._image_dir / split if self._has_image_split else self._image_dir
            split_value = split if self._has_image_split else "main"

            with open(self._annotation_dir / f"{split}.json", "r") as f:
                json_data = json.load(f)

            total_data = len(json_data)
            if total_data == 0:
                raise "annotation file is empty"

            for data in json_data:
                image_name = data["image"]
                image_path = image_dir / image_name
                # check if image file exists in the image directory
                if not image_path.is_file():
                    warnings.warn(f"Not able to find image {image_name} in path {image_dir}.")
                    continue
                image_width, image_height = imagesize.get(image_path)
                for annotation in data["annotations"]:
                    master_data["image_id"].append(image_name)
                    master_data["image_path"].append(image_dir.joinpath(image_name))
                    master_data["width"].append(annotation["coordinates"]["width"])
                    master_data["height"].append(annotation["coordinates"]["height"])
                    master_data["x_min"].append(annotation["coordinates"]["x"])
                    master_data["y_min"].append(annotation["coordinates"]["y"])
                    master_data["category"].append(annotation["label"])
                    master_data["image_height"].append(image_height)
                    master_data["image_width"].append(image_width)
                    master_data["split"].append(split_value)

        df = pd.DataFrame(master_data)
        # creating class ids based on unique categories
        class_map_df = df["category"].drop_duplicates().reset_index(drop=True).to_frame()
        class_map_df["class_id"] = class_map_df.index.values
        self.master_df = pd.merge(df, class_map_df, on="category")
