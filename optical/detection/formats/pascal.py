"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import itertools
import xml.etree.ElementTree as ET
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from .base import FormatSpec
from .utils import (NUM_THREADS, Pathlike, exists, get_annotation_dir,
                    get_image_dir)


class Pascal(FormatSpec):
    """Represents a Pascal annotation object.

    Args:
        root (Pathlike): path to root directory. Expects the ``root`` directory to have either
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
                    |   ├── 1.xml
                    │   ├── 2.xml
                    │   │   ...
                    │   └── n.xml
                    ├── valid (...)
                    └── test (...)

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
                    ├── 1.xml
                    ├── 2.xml
                    │   ...
                    └── n.xml

    """

    def __init__(self, root: Pathlike):
        super().__init__(root)
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._find_splits()
        self._resolve_dataframe()

    def read_xml(self, xml_filepath: Pathlike):
        """parses a VOC annotation xml file"""
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        class_labels = [obj.find("name").text for obj in root.findall("object")]
        x_mins = [int(obj.find("bndbox").find("xmin").text) for obj in root.findall("object")]
        y_mins = [int(obj.find("bndbox").find("ymin").text) for obj in root.findall("object")]

        box_widths = [
            int(obj.find("bndbox").find("xmax").text) - int(obj.find("bndbox").find("xmin").text)
            for obj in root.findall("object")
        ]
        box_heights = [
            int(obj.find("bndbox").find("ymax").text) - int(obj.find("bndbox").find("ymin").text)
            for obj in root.findall("object")
        ]

        img_filenames = [root.find("filename").text] * len(class_labels)
        img_widths = [root.find("size").find("width").text] * len(class_labels)
        img_heights = [root.find("size").find("height").text] * len(class_labels)
        return img_filenames, img_heights, img_widths, class_labels, x_mins, y_mins, box_widths, box_heights

    def read_xmls(self, path):
        """read Pascal VOC annotation xml files"""
        xml_files = list(Path(path).glob("*.xml"))

        labels = Parallel(n_jobs=NUM_THREADS, backend="threading")(
            delayed(self.read_xml)(file_path) for file_path in tqdm(xml_files)
        )

        img_filenames = list(itertools.chain(*map(itemgetter(0), labels)))
        img_heights = list(itertools.chain(*map(itemgetter(1), labels)))
        img_widths = list(itertools.chain(*map(itemgetter(2), labels)))
        class_labels = list(itertools.chain(*map(itemgetter(3), labels)))
        x_mins = list(itertools.chain(*map(itemgetter(4), labels)))
        y_mins = list(itertools.chain(*map(itemgetter(5), labels)))
        box_widths = list(itertools.chain(*map(itemgetter(6), labels)))
        box_heights = list(itertools.chain(*map(itemgetter(7), labels)))

        records = pd.DataFrame(
            dict(
                image_id=img_filenames,
                image_height=img_heights,
                image_width=img_widths,
                category=class_labels,
                x_min=x_mins,
                y_min=y_mins,
                width=box_widths,
                height=box_heights,
            )
        )
        return records

    def _resolve_dataframe(self):
        if self._has_image_split:
            label_df = pd.DataFrame(
                columns=[
                    "image_id",
                    "image_height",
                    "image_width",
                    "category",
                    "x_min",
                    "y_min",
                    "width",
                    "height",
                    "split",
                    "image_path",
                ]
            )

            for split in self._splits:
                annotation_dir = self._annotation_dir / f"{split}"
                image_dir = self._image_dir / f"{split}"

                split_df = self.read_xmls(annotation_dir)
                split_df["split"] = split
                split_df["image_path"] = split_df["image_id"].map(lambda x: Path(image_dir).joinpath(x))
                label_df = pd.concat([label_df, split_df], ignore_index=True)

        else:
            label_df = self.read_xmls(self._annotation_dir)
            label_df["split"] = "main"
            label_df["image_path"] = label_df["image_id"].map(lambda x: Path(self._image_dir).joinpath(x))

        lbl = LabelEncoder()
        label_df["class_id"] = lbl.fit_transform(label_df["category"])

        for col in ["x_min", "y_min", "width", "height"]:
            label_df[col] = label_df[col].astype(np.float32)
        for col in ["image_width", "image_height"]:
            label_df[col] = label_df[col].astype(np.int32)

        self.master_df = label_df
