"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""


import os
import warnings
from pathlib import Path
from typing import Union

import imagesize
import yaml
import numpy as np
import pandas as pd

from .base import FormatSpec
from .utils import exists, get_image_dir, get_annotation_dir


class Yolo(FormatSpec):
    """Represents a YOLO annotation object.

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
                    ├── train
                    │   ├── 1.txt
                    │   ├── 2.txt
                    │   │   ...
                    │   └── n.txt
                    ├── valid (...)
                    ├── test (...)
                    └── dataset.yaml [Optional]

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
                    ├── 1.txt
                    ├── 2.txt
                    │ ...
                    ├── n.txt
                    └── dataset.yaml [Optional]
    """

    def __init__(self, root: Union[str, os.PathLike]):
        # self.root = root
        super().__init__(root)
        self.class_file = [y for y in Path(self.root).glob("*.yaml")]
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing 'images' directory."
        assert exists(self._annotation_dir), "root is missing 'annotations' directory."
        self._find_splits()
        self._resolve_dataframe()

    def _resolve_dataframe(self):

        master_df = pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "image_width",
                "image_height",
                "x_min",
                "y_min",
                "width",
                "height",
                "category",
                "image_path",
            ],
        )

        for split in self._splits:
            image_ids = []
            image_paths = []
            class_ids = []
            x_mins = []
            y_mins = []
            bbox_widths = []
            bbox_heights = []
            image_heights = []
            image_widths = []

            split = split if self._has_image_split else ""
            annotations = Path(self._annotation_dir).joinpath(split).glob("*.txt")

            for txt in annotations:
                stem = txt.stem
                try:
                    img_file = list(Path(self._image_dir).joinpath(split).glob(f"{stem}*"))[0]
                    im_width, im_height = imagesize.get(img_file)
                    with open(txt, "r") as f:
                        instances = f.read().strip().split("\n")
                        for ins in instances:
                            class_id, x, y, w, h = list(map(float, ins.split()))
                            image_ids.append(img_file.name)
                            image_paths.append(img_file)
                            class_ids.append(int(class_id))
                            x_mins.append(max(float((float(x) - w / 2) * im_width), 0))
                            y_mins.append(max(float((y - h / 2) * im_height), 0))
                            bbox_widths.append(float(w * im_width))
                            bbox_heights.append(float(h * im_height))
                            image_widths.append(im_width)
                            image_heights.append(im_height)

                except IndexError:  # if the image file does not exist
                    pass

            annots_df = pd.DataFrame(
                list(
                    zip(
                        image_ids,
                        image_paths,
                        image_widths,
                        image_heights,
                        class_ids,
                        x_mins,
                        y_mins,
                        bbox_widths,
                        bbox_heights,
                    )
                ),
                columns=[
                    "image_id",
                    "image_path",
                    "image_width",
                    "image_height",
                    "class_id",
                    "x_min",
                    "y_min",
                    "width",
                    "height",
                ],
            )
            annots_df["split"] = split if split else "main"
            master_df = pd.concat([master_df, annots_df], ignore_index=True)

        # get category names from `dataset.yaml`
        try:
            with open(Path(self._annotation_dir).joinpath("dataset.yaml")) as f:
                label_desc = yaml.load(f, Loader=yaml.FullLoader)

            categories = label_desc["names"]
            label_map = dict(zip(range(len(categories)), categories))
        except FileNotFoundError:
            label_map = dict()
            warnings.warn(f"No `dataset.yaml` file found in {self._annotation_dir}")

        master_df["class_id"] = master_df["class_id"].astype(np.int32)

        if label_map:
            master_df["category"] = master_df["class_id"].map(label_map)
        else:
            master_df["category"] = master_df["class_id"].astype(str)
        self.master_df = master_df
