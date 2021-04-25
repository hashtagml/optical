"""
__author__: HashTagML
license: MIT
Created: Friday, 16th April 2021
"""

import os
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


from .base import FormatSpec
from .utils import _tf_parse_example, tf_decode_image


class Tfrecord(FormatSpec):
    """Represents a tfrecord annotation object.

    Args:
        root (Union[str, os.PathLike]): path to root directory. Expects the ``root`` directory to have either
           of the following layouts:

           .. code-block:: bash

                root
                ├──train.tfrecord
                ├──test.tfrecord
                ├──valid.tfrecord

    """

    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self._has_image_split = False
        image_path = Path(root) / "images"
        image_path.mkdir(parents=True, exist_ok=True)
        self._annotation_dir = root
        self._image_dir = image_path
        self._splits = self._find_splits()
        self._resolve_dataframe()

    def _find_splits(self):
        ann_splits = [x.stem for x in Path(self.root).glob("*.tfrecord")]

        assert ann_splits, "directory doesn't have tfrecords"
        if len(ann_splits) > 1:
            self._has_image_split = True

        return ann_splits

    def _resolve_dataframe(self):
        import tensorflow as tf

        img_filenames = []
        img_widths = []
        img_heights = []
        cls_names = []
        x_mins = []
        y_mins = []
        box_widths = []
        box_heights = []
        splits = []
        cls_ids = []
        img_paths = []
        for split in self._splits:
            tf_record = str(Path(self.root) / f"{split}.tfrecord")
            render = tf.data.TFRecordDataset(tf_record)
            dataset = render.map(_tf_parse_example)
            img_dir = Path(self.root) / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            for data in dataset:
                img_filename = data["image/filename"].numpy().decode("utf-8")
                img_height = int(data["image/height"].numpy())
                img_width = int(data["image/width"].numpy())
                bbox_len = data["image/object/bbox/xmin"].shape[0]
                for i in range(bbox_len):
                    xmin = data["image/object/bbox/xmin"].values[i].numpy() * img_width
                    ymin = data["image/object/bbox/ymin"].values[i].numpy() * img_height
                    x_mins.append(xmin)
                    y_mins.append(ymin)
                    box_width = data["image/object/bbox/xmax"].values[i].numpy() * img_width - xmin
                    box_height = data["image/object/bbox/ymax"].values[i].numpy() * img_height - ymin
                    box_widths.append(box_width)
                    box_heights.append(box_height)
                    cls_names.append(data["image/object/class/text"].values[i].numpy().decode("utf-8"))
                    img_filenames.append(img_filename)
                    img_heights.append(img_height)
                    img_widths.append(img_width)
                    cls_ids.append(data["image/object/class/label"].values[i].numpy())
                    splits.append(split)
                    img_paths.append(self._image_dir.joinpath(split, img_filename))
            _ = Parallel(n_jobs=-1, backend="threading")(
                delayed(tf_decode_image)(self.root, data, split) for data in dataset
            )
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
                    cls_ids,
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
