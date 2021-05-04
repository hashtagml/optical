"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import os


from typing import Union
import numpy as np

import pandas as pd

from .base import FormatSpec
from .utils import exists, get_image_dir, get_annotation_dir, read_xml


class Pascal(FormatSpec):
    """Represents a Pascal annotation object.

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
        if self._has_image_split:
            img_filenames = []
            img_widths = []
            img_heights = []
            cls_names = []
            x_mins = []
            y_mins = []
            box_widths = []
            box_heights = []
            splits = []
            img_paths = []

            for split in self._splits:
                folder_files = self._annotation_dir / f"{split}"
                img_folder = self._image_dir / f"{split}"
                (
                    s_img_names,
                    s_img_widths,
                    s_img_heights,
                    s_cls_names,
                    s_x_mins,
                    s_y_mins,
                    s_box_widths,
                    s_box_heights,
                    s_img_paths,
                ) = read_xml(folder_files, img_folder)
                s_splits = [split for i in range(len(s_img_names))]
                img_filenames.extend(s_img_names)
                img_widths.extend(s_img_widths)
                img_heights.extend(s_img_heights)
                cls_names.extend(s_cls_names)
                x_mins.extend(s_x_mins)
                y_mins.extend(s_y_mins)
                box_widths.extend(s_box_widths)
                box_heights.extend(s_box_heights)
                splits.extend(s_splits)
                img_paths.extend(s_img_paths)
        else:
            (
                img_filenames,
                img_widths,
                img_heights,
                cls_names,
                x_mins,
                y_mins,
                box_widths,
                box_heights,
                img_paths,
            ) = read_xml(self._annotation_dir, self._image_dir)
            splits = ["main" for i in range(len(img_filenames))]

        class_dict = dict(zip(set(cls_names), [i for i in range(1, len(set(cls_names)) + 1)]))
        class_ids = [class_dict[cate] for cate in cls_names]
        master_df = pd.DataFrame(
            list(
                zip(
                    img_filenames,
                    img_widths,
                    img_heights,
                    x_mins,
                    y_mins,
                    box_widths,
                    box_heights,
                    cls_names,
                    class_ids,
                    splits,
                    img_paths,
                )
            ),
            columns=[
                "image_id",
                "image_width",
                "image_height",
                "x_min",
                "y_min",
                "width",
                "height",
                "category",
                "class_id",
                "split",
                "image_path",
            ],
        )
        for col in ["x_min", "y_min", "width", "height"]:
            master_df[col] = master_df[col].astype(np.float32)
        for col in ["image_width", "image_height"]:
            master_df[col] = master_df[col].astype(np.int32)

        self.master_df = master_df
