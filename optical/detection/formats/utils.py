"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import json
import os
import shutil
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed

NUM_THREADS = os.cpu_count() // 2
_TF_INSTALLED = True

try:
    import tensorflow as tf
except ImportError:
    _TF_INSTALLED = False

Pathlike = Union[str, Path]


class DetectionFormat(str, Enum):
    COCO = "coco"
    CREATEML = "createml"
    CSV = "csv"
    PASCAL_VOC = "pascal_voc"
    SAGEMAKER_MANIFEST = "sagemaker_manifest"
    JSON = "json"
    TFRECORD = "tfrecord"
    YOLO = "yolo"


file_extensions = dict()
file_extensions[DetectionFormat.COCO] = "json"
file_extensions[DetectionFormat.CREATEML] = "json"
file_extensions[DetectionFormat.CSV] = "csv"
file_extensions[DetectionFormat.PASCAL_VOC] = "xml"
file_extensions[DetectionFormat.SAGEMAKER_MANIFEST] = "manifest"
file_extensions[DetectionFormat.JSON] = "json"
file_extensions[DetectionFormat.TFRECORD] = "tfrecord"
file_extensions[DetectionFormat.YOLO] = "txt"


class CopyType(str, Enum):
    SOFT = "soft"
    HARD = "hard"


def ifnone(x: Any, y: Any, transform: Optional[Callable] = None, type_safe: bool = False):
    """if x is None return y otherwise x after applying transofrmation ``transform`` and
    casting the result back to original type if ``type_safe``

    Args:
        x (Any): returns x if x is not none
        y (Any): returns y if x is none
        transform (Optional[Callable], optional): applies transform to the output. Defaults to None.
        type_safe (bool, optional): if true, tries casting the output to the original type. Defaults to False.
    """

    if transform is not None:
        assert callable(transform), "`transform` should be either `None` or instance of `Callable`"
    else:

        def transform(x):
            return x

    if x is None:
        orig_type = type(y)
        out = transform(y)
    else:
        orig_type = type(x)
        out = transform(x)
    if type_safe:
        try:
            out = orig_type(out)
        except (ValueError, TypeError):
            warnings.warn(f"output could not be casted as type {orig_type.__name__}")
    return out


def exists(path: Pathlike):
    """checks for whether a directory or file exists in the specified path"""
    if Path(path).is_dir():
        return "dir"

    if Path(path).is_file():
        return "file"

    return


def get_image_dir(root: Pathlike) -> Pathlike:
    """returns image directory given a root directory"""
    return Path(root) / "images"


def get_annotation_dir(root: Pathlike) -> Pathlike:
    """returns annotation directory given a root directory"""
    return Path(root) / "annotations"


def write_json(data_dict: Dict, filename: Pathlike) -> None:
    """writes json to disk"""
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2)


def partiion_df(df: pd.DataFrame, split: Optional[str] = None, category: Optional[str] = None) -> pd.DataFrame:
    """partition the annotation dataframe by split and/or label category

    Args:
        df (pd.DataFrame): the  dataframe.
        split (Optional[str], optional): the dataset split e.g., ``train``, ``test`` etc. Defaults to None.
        category (Optional[str], optional): the label category. Defaults to None.

    Raises:
        ValueError: if an unknown category is specified.

    Returns:
        pd.DataFrame: the partitioned dataframe.
    """
    if split is not None:
        df = df.query("split == @split").copy()

    if category is not None:
        if category not in df.category.unique():
            raise ValueError(f"class `{category}` is not present in annotations")
        df = df.query("category == @category").copy()

    return df


def get_class_id_category_mapping(df: pd.DataFrame) -> Dict[int, str]:
    """returns the class_id to category mapping from the annotation dataframe

    Args:
        df (pd.DataFrame): the annotation dataframe

    Returns:
        Dict[int, str]: class indices to class labels mapping
    """
    set_df = df.drop_duplicates(subset="class_id")[["category", "class_id"]]
    return set_df.set_index("class_id")["category"].to_dict()


def find_splits(image_dir: Pathlike, annotation_dir: Pathlike, format: DetectionFormat) -> Tuple[List[str], bool]:
    """discover the splits in the dataset, will ignore splits that have no annotation(s)"""

    # check if the image directory has internal splits
    im_splits = [x.name for x in Path(image_dir).iterdir() if x.is_dir() and not x.name.startswith(".")]

    # per image annotation formats
    if format in (DetectionFormat.YOLO, DetectionFormat.PASCAL_VOC):
        ann_splits = [x.name for x in Path(annotation_dir).iterdir() if x.is_dir()]

        if not ann_splits:  # if flat directory
            files = list(Path(annotation_dir).glob(f"*.{file_extensions[format]}"))
            if len(files):
                ann_splits = ["main"]
            else:
                raise ValueError(f"No annotation found in {annotation_dir}. Please check the directory layout.")

    # consolidated file fornats e.g., ``coco``, ``csv`` etc.
    else:
        ann_splits = [x.stem for x in Path(annotation_dir).glob(f"*.{file_extensions[format]}")]

    no_anns = set(im_splits).difference(ann_splits)
    if no_anns:
        warnings.warn(f"no annotation found for {', '.join(list(no_anns))}")

    return ann_splits, len(im_splits) > 0


def _tf_parse_example(example):
    """parse tf examples"""
    features = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example, features)


def copy_file(
    src: Pathlike, dest: Pathlike, filename: Optional[Pathlike] = None, type: CopyType = CopyType.SOFT
) -> None:
    """copies a file from one directory to another

    Args:
        src (Pathlike): either a directory containing files or any filepath.
        dest (Pathlike): the output directory for the copy.
        filename (Optional[Union[str, os.PathLike]], optional): If ``src`` is a directory, the name of the
           file to copy. Defaults to None.
        hard (bool): Whether to physically copy the file (hard) or create a symbolic link (soft). Defaults to False.
    """

    filename = Path(src).joinpath(filename) if filename else Path(src)
    dest = Path(dest) / filename.name

    if filename.is_file():
        if type == CopyType.HARD:
            shutil.copyfile(filename, dest)
        else:
            dest.symlink_to(filename)


def copy_files(src_files: List[Pathlike], dest_dir: Pathlike, type: CopyType = CopyType.SOFT) -> None:

    # NOTE: joblib is slower than serial for symlink creation
    if type == CopyType.HARD:
        _ = Parallel(n_jobs=-1, backend="threading")(
            delayed(copy_file)(f, dest_dir, None, CopyType.HARD) for f in src_files
        )
    else:
        for file in src_files:
            copy_file(file, dest_dir, type=CopyType.SOFT)


def makedirs(src: Pathlike, ext: str, dest: Optional[Pathlike] = None) -> None:
    """creates necessary directories"""
    output_dir = ifnone(dest, src, Path)
    output_dir = output_dir / ext
    output_imagedir = output_dir / "images"
    output_labeldir = output_dir / "annotations"
    output_imagedir.mkdir(parents=True, exist_ok=True)
    output_labeldir.mkdir(parents=True, exist_ok=True)
    return output_imagedir, output_labeldir
