"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""


import os
import warnings
from pathlib import Path
from typing import Dict, Union

import imagesize
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from functools import partial

from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir, NUM_THREADS


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

            parse_partial = partial(self._parse_txt_file, split)
            print("Loading yolo annotations:")
            all_instances = Parallel(n_jobs=NUM_THREADS, backend="threading")(
                delayed(parse_partial)(txt) for txt in tqdm(annotations, desc=split)
            )
            for instances in all_instances:
                image_ids.extend(instances["image_ids"])
                image_paths.extend(instances["image_paths"])
                class_ids.extend(instances["class_ids"])
                x_mins.extend(instances["x_mins"])
                y_mins.extend(instances["y_mins"])
                bbox_widths.extend(instances["bbox_widths"])
                bbox_heights.extend(instances["bbox_heights"])
                image_widths.extend(instances["image_widths"])
                image_heights.extend(instances["image_heights"])

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

    def _parse_txt_file(self, split: str, txt: Union[str, os.PathLike]) -> Dict:
        """Parse txt annotations in yolo format

        Args:
            split (str): dataset split
            txt (Union[str, os.PathLike]): annotations file path

        Returns:
            Dict: dict containing scaled annotation for each line in the text file.
        """
        label_info_keys = [
            "image_ids",
            "image_paths",
            "class_ids",
            "x_mins",
            "y_mins",
            "bbox_widths",
            "bbox_heights",
            "image_heights",
            "image_widths",
        ]
        label_info = {key: [] for key in label_info_keys}
        stem = txt.stem
        try:
            img_file = list(Path(self._image_dir).joinpath(split).glob(f"{stem}*"))[0]
            im_width, im_height = imagesize.get(img_file)
        except IndexError:  # if the image file does not exist
            return label_info

        with open(txt, "r") as f:
            instances = f.read().strip().split("\n")
            for ins in instances:
                class_id, x, y, w, h = list(map(float, ins.split()))
                label_info["image_ids"].append(img_file.name)
                label_info["image_paths"].append(img_file)
                label_info["class_ids"].append(int(class_id))
                label_info["x_mins"].append(max(float((float(x) - w / 2) * im_width), 0))
                label_info["y_mins"].append(max(float((y - h / 2) * im_height), 0))
                label_info["bbox_widths"].append(float(w * im_width))
                label_info["bbox_heights"].append(float(h * im_height))
                label_info["image_widths"].append(im_width)
                label_info["image_heights"].append(im_height)
        return label_info
