"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import json
import os
from typing import Union

import pandas as pd

from .base import FormatSpec
from .utils import exists, find_job_metadata_key, get_annotation_dir, get_image_dir


class SageMaker(FormatSpec):
    """Class to handle sagemaker '.manifest' annotation transformations

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
                    ├── train.manifest
                    ├── valid.manifest
                    └── test.manifest

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
                    └── label.manifest
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
            "class_id": [],
            "category": [],
            "split": [],
        }
        for split in self._splits:
            image_dir = self._image_dir / split if self._has_image_split else self._image_dir
            split_value = split if self._has_image_split else "main"

            with open(self._annotation_dir / f"{split}.manifest") as f:
                manifest_lines = f.readlines()

            total_data = len(manifest_lines)
            if total_data == 0:
                raise "input file is empty"

            for line in manifest_lines:
                json_line = json.loads(line)
                job_metadata_key = find_job_metadata_key(json_line)
                assert (
                    json_line[job_metadata_key]["type"] == "groundtruth/object-detection"
                ), "supports object detection manifest files"

                class_map = json_line[job_metadata_key]["class-map"]
                job_name = json_line[job_metadata_key]["job-name"].split("/")[-1]
                for annotation in json_line[job_name]["annotations"]:
                    img_name = json_line["source-ref"].split("/")[-1]
                    master_data["image_id"].append(img_name)
                    master_data["image_path"].append(image_dir.joinpath(img_name))
                    master_data["image_height"].append(json_line[job_name]["image_size"][0]["height"])
                    master_data["image_width"].append(json_line[job_name]["image_size"][0]["width"])
                    master_data["width"].append(annotation["width"])
                    master_data["height"].append(annotation["height"])
                    master_data["x_min"].append(annotation["left"])
                    master_data["y_min"].append(annotation["top"])
                    master_data["class_id"].append(str(annotation["class_id"]))
                    master_data["category"].append(class_map[str(annotation["class_id"])])
                    master_data["split"].append(split_value)
        self.master_df = pd.DataFrame(master_data)
