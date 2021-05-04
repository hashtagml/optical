"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""


import json
import os
from pathlib import Path
from typing import Union

import imagesize
import pandas as pd

from .base import FormatSpec
from .utils import exists, get_annotation_dir, get_image_dir


class SimpleJson(FormatSpec):
    """Represents a SimleJson annotation object.

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
            "score",
        ]
        image_ids, image_paths, image_widths, image_heights = [], [], [], []
        x_mins, y_mins, widths, heights = [], [], [], []
        categorys, class_ids, scores, splits = [], [], [], []

        for split in self._splits:

            simple_json = self._annotation_dir / f"{split}.json"
            with open(simple_json) as f:
                annotations = json.load(f)
            class_map = {}

            num_images = len(annotations)
            num_anns = 0
            if num_images == 0:
                raise RuntimeWarning(f"Annotation file {simple_json} is empty. Please check.")

            for im_id, anns in annotations.items():

                split_path = split if self._has_image_split else ""
                im_path = list(Path(self._image_dir).joinpath(split_path).glob(f"{im_id}"))[0]
                im_width, im_height = imagesize.get(im_path)
                if not len(anns):
                    image_ids.append(im_id)
                    image_paths.append(im_path)
                    image_widths.append(im_width)
                    image_heights.append(im_height)
                    x_mins.append(None), y_mins.append(None), widths.append(None), heights.append(None)
                    categorys.append(None), class_ids.append(None), scores.append(None)
                    splits.append(split)
                for ann in anns:
                    image_ids.append(im_id)
                    image_paths.append(im_path)
                    image_widths.append(im_width)
                    image_heights.append(im_height)
                    bbox = ann["bbox"]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    x_mins.append(bbox[0]), y_mins.append(bbox[1]), widths.append(bbox[2]), heights.append(bbox[3])
                    category = ann["classname"]
                    categorys.append(category)
                    if class_map.get(category, None) is None:
                        class_id = len(class_map)
                        class_map[category] = class_id
                        class_ids.append(class_id)
                    else:
                        class_ids.append(class_map[category])
                    scores.append(ann.get("confidence", None))
                    splits.append(split)
                    num_anns += 1
        data = {}
        for col in columns:
            data[col] = eval(col + "s")
        self.master_df = pd.DataFrame(data=data)
