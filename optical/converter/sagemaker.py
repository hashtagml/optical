"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import json
import os
import warnings
from pathlib import Path
from typing import Union

import pandas as pd

from .base import FormatSpec
from .utils import exists, find_job_metadata_key, get_annotation_dir, get_image_dir


class SageMaker(FormatSpec):
    """Class to handle sagemaker '.manifest' annotation transformations

    Args:
        FormatSpec : Base class to inherit class attributes
    """

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
        ann_splits = [x.stem for x in Path(self._annotation_dir).glob("*.manifest")]

        if im_splits:
            self._has_image_split = True

        if not len(ann_splits):
            warnings.warn(f"no annotation file found inside {self._annotation_dir}")

        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {', '.join(list(no_anns))}")
        return ann_splits

    def _resolve_dataframe(self):
        master_data = {
            "image_id": [],
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

            with open(os.path.join(self._annotation_dir, f"{split}.manifest")) as f:
                manifest_lines = f.readlines()

            if len(manifest_lines) == 0:
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
                    master_data["image_id"].append(json_line["source-ref"].split("/")[-1])
                    master_data["image_height"].append(json_line[job_name]["image_size"][0]["height"])
                    master_data["image_width"].append(json_line[job_name]["image_size"][0]["width"])
                    master_data["width"].append(annotation["width"])
                    master_data["height"].append(annotation["height"])
                    master_data["x_min"].append(annotation["left"])
                    master_data["y_min"].append(annotation["top"])
                    master_data["class_id"].append(str(annotation["class_id"]))
                    master_data["category"].append(class_map[str(annotation["class_id"])])
                    master_data["split"].append(split)
        self.master_df = pd.DataFrame(master_data)
