"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""

from typing import Union, List, Dict
import os
from pathlib import Path
import warnings
from joblib import Parallel, delayed
import pandas as pd

from ..utils import exists, get_image_dir, get_annotation_dir, read_coco


def extract_info(annot: Dict):
    return annot["category_id"], annot["bbox"]


class COCO:
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._splits = self._find_splits()

    def _find_splits(self):
        im_splits = [x.name for x in Path(self._image_dir).iterdir() if x.is_dir()]
        ann_splits = [x.stem for x in Path(self._annotation_dir).glob("*.json")]
        print(im_splits)
        print(ann_splits)
        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {', '.join(list(no_anns))}")
        return ann_splits

    def _get_class_map(self, categories: List):
        class_map = dict()
        for cat in categories:
            class_map[cat["id"]] = cat["name"]
        return class_map

    def describe(self, classwise: bool = False):
        split_str = []
        for split in self._splits:
            coco_json = self._annotation_dir / f"{split}.json"
            images, annots, cats = read_coco(coco_json)
            split_str.append([len(images), len(annots), len(cats)])

            # if classwise:
            class_map = self._get_class_map(cats)
            info = Parallel(n_jobs=-1, backend="threading")(delayed(extract_info)(ann) for ann in annots)
            # a, b = [], []
            # for ann in annots:
            #     a.append(ann["category_id"])
            #     b.append(ann["bbox"])

            # df = pd.DataFrame({"class_id": a, "bbox": b})
            df = pd.DataFrame(info, columns=["class_id", "bbox"])
            setattr(self, split, df)
