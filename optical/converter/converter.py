"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""
# TODO: needs better solution for Handling TFrecords

import copy
import json
import os
import warnings
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .utils import (
    copyfile,
    get_id_to_class_map,
    ifnone,
    write_json,
    write_xml,
    create_tf_example,
    write_label_map,
)


class LabelEncoder:
    def __init__(self):
        self._map = dict()

    def fit(self, series):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        categories = series.unique().tolist()
        label_map = dict(zip(categories, np.arange(len(categories))))
        for k, _ in label_map.items():
            if k not in self._map:
                self._map[k] = label_map[k]

    def transform(self, series):
        series = series.map(self._map)
        return series

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)


def _fastcopy(src_files: Union[str, os.PathLike], dest_dir: Union[str, os.PathLike]):
    _ = Parallel(n_jobs=-1, backend="threading")(delayed(copyfile)(f, dest_dir) for f in src_files)


def write_yolo_txt(filename: str, output_dir: Union[str, os.PathLike, PosixPath], yolo_string: str):
    filepath = Path(output_dir).joinpath(Path(filename).stem + ".txt")
    with open(filepath, "a") as f:
        f.write(yolo_string)
        f.write("\n")


def _makedirs(src: Union[str, os.PathLike], ext: str, dest: Optional[Union[str, os.PathLike]] = None):
    output_dir = ifnone(dest, src, Path)
    output_dir = output_dir / ext
    output_imagedir = output_dir / "images"
    output_labeldir = output_dir / "annotations"
    output_imagedir.mkdir(parents=True, exist_ok=True)
    output_labeldir.mkdir(parents=True, exist_ok=True)
    return output_imagedir, output_labeldir


def convert_yolo(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """converts to yolo from master dataframe

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
        save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
            annotation. Defaults to ``None``.
    """

    save_under = ifnone(save_under, "yolo")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    lbl = LabelEncoder()

    dataset = dict()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 1 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()

        # drop images missing width or height information
        hw_missing = split_df[pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])]

        split_df["x_center"] = split_df["x_min"] + split_df["width"] / 2
        split_df["y_center"] = split_df["y_min"] + split_df["height"] / 2

        # normalize
        split_df["x_center"] = split_df["x_center"] / split_df["image_width"]
        split_df["y_center"] = split_df["y_center"] / split_df["image_height"]
        split_df["width"] = split_df["width"] / split_df["image_width"]
        split_df["height"] = split_df["height"] / split_df["image_height"]

        split_df["class_index"] = lbl.fit_transform(split_df["category"])

        split_df["yolo_string"] = (
            split_df["class_index"].astype(str)
            + " "
            + split_df["x_center"].astype(str)
            + " "
            + split_df["y_center"].astype(str)
            + " "
            + split_df["width"].astype(str)
            + " "
            + split_df["height"].astype(str)
        )

        ds = split_df.groupby("image_id")["yolo_string"].agg(lambda x: "\n".join(x)).reset_index()

        image_ids = ds["image_id"].tolist()
        yolo_strings = ds["yolo_string"].tolist()

        dataset[split] = str(Path(root) / "images" / split)

        for image_id, ystr in tqdm(zip(image_ids, yolo_strings), total=len(image_ids), desc=f"split: {split}"):
            write_yolo_txt(image_id, output_subdir, ystr)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)

    dataset["nc"] = len(lbl._map)
    dataset["names"] = list(lbl._map.keys())

    with open(Path(output_labeldir).joinpath("dataset.yaml"), "w") as f:
        yaml.dump(dataset, f, default_flow_style=None, allow_unicode=True)


def convert_csv(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):

    save_under = ifnone(save_under, "csv")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    df = copy.deepcopy(df)
    df["x_max"] = df["x_min"] + df["width"]
    df["y_max"] = df["y_min"] + df["height"]
    df.drop(["width", "height"], axis=1, inplace=True)
    for col in ("x_min", "y_min", "x_max", "y_max"):
        df[col] = df[col].astype(np.int32)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()
        split_df.drop(["split"], axis=1, inplace=True)

        image_paths = split_df["image_path"].unique().tolist()
        split_df = split_df[
            ["image_id", "image_width", "image_height", "x_min", "y_min", "x_max", "y_max", "category"]
        ]
        split_df.to_csv(output_labeldir.joinpath(f"{split}.csv"), index=False)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(image_paths, dest_dir)


def _make_coco_images(df: pd.DataFrame, image_map: Dict) -> List:
    """makes images list for coco"""
    df = copy.deepcopy(df)
    df.drop_duplicates(subset=["image_id"], keep="first", inplace=True)
    df = (
        df[["image_id", "image_height", "image_width"]]
        .copy()
        .rename(columns={"image_id": "file_name", "image_height": "height", "image_width": "width"})
    )
    df["id"] = df["file_name"].map(image_map)
    df = df[["id", "file_name", "height", "width"]]
    image_list = list(df.to_dict(orient="index").values())
    return image_list


def _make_coco_annotations(df: pd.DataFrame, image_map: Dict) -> List:
    """makes annotation list for coco"""

    df = copy.deepcopy(df)
    df["bbox"] = df[["x_min", "y_min", "width", "height"]].apply(list, axis=1)
    df["area"] = df["height"] * df["width"]
    df.drop(["x_min", "y_min", "width", "height", "image_width", "image_height"], axis=1, inplace=True)
    df["id"] = range(len(df))
    df["image_id"] = df["image_id"].map(image_map)
    df.rename(columns={"class_id": "category_id"}, inplace=True)
    df["category_id"] = df["category_id"].astype(int)
    df["segmentation"] = [[]] * len(df)
    df["iscrowd"] = 0
    df = df[["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"]].copy()
    annotation_list = list(df.to_dict(orient="index").values())
    return annotation_list


def _make_coco_categories(df: pd.DataFrame) -> List:
    """makes category list for coco"""

    df = copy.deepcopy(df)
    df = (
        df.drop_duplicates(subset=["category"], keep="first")
        .sort_values("class_id")[["class_id", "category"]]
        .rename(columns={"category": "name", "class_id": "id"})
    )
    df["id"] = df["id"].astype(int)
    df["supercategory"] = "none"
    category_list = list(df.to_dict(orient="index").values())
    return category_list


def convert_coco(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """converts to coco from master df

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
        save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
            annotation. Defaults to ``None``.
    """

    save_under = ifnone(save_under, "coco")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()
        images = df["image_id"].unique().tolist()

        image_map = dict(zip(images, range(len(images))))
        image_list = _make_coco_images(split_df, image_map)
        annotation_list = _make_coco_annotations(split_df, image_map)
        category_list = _make_coco_categories(split_df)

        coco_dict = dict()
        coco_dict["images"] = image_list
        coco_dict["annotations"] = annotation_list
        coco_dict["categories"] = category_list

        output_file = output_labeldir / f"{split}.json"
        # print(output_file)
        write_json(coco_dict, output_file)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def _make_manifest_data(image_info: List, grouped_info: pd.DataFrame, job_name: str, id_to_class_map: Dict):
    # creating json like data for each row of df
    manifest_dic = {}
    manifest_dic["source-ref"] = image_info[0]
    manifest_dic[f"{job_name}"] = {
        "image_size": [{"width": int(image_info[1]), "depth": 3, "height": int(image_info[2])}],
    }
    manifest_dic[f"{job_name}-metadata"] = {
        "job-name": job_name,
        "class-map": id_to_class_map,
        "creation-date": str(datetime.now()),
        "type": "groundtruth/object-detection",
    }
    annotations = grouped_info[["class_id", "height", "width", "top", "left"]].to_dict("records")
    # append annotations
    manifest_dic[f"{job_name}"]["annotations"] = annotations

    return manifest_dic


def convert_sagemaker(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
    job_name: str = "optical",
):
    """converts to sagemaker .manifest from master dataframe

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
        save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
            annotation. Defaults to ``None``.
        job_name(Optional[str]): manifest job name for the output file. Defaults to optical
    """

    save_under = ifnone(save_under, "sagemaker")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    for split in splits:
        # if split == "main":
        #     output_subdir = output_labeldir
        # else:
        #     output_subdir = output_labeldir / split
        # output_subdir.mkdir(parents=True, exist_ok=True)
        split_df = df.query("split == @split").copy()

        # drop images missing width or height information
        hw_missing = split_df[pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])]
        split_df = split_df.rename(columns={"y_min": "top", "x_min": "left"})
        id_to_class_map = get_id_to_class_map(split_df)
        grouped_split_df = split_df.groupby(["image_id", "image_width", "image_height"])

        with open(output_labeldir / f"{split}.manifest", "w") as f:
            for image_info, grouped_info in tqdm(
                grouped_split_df, total=grouped_split_df.ngroups, desc=f"split: {split}"
            ):
                manifest_dic = _make_manifest_data(image_info, grouped_info, job_name, id_to_class_map)
                f.write(json.dumps(manifest_dic) + "\n")

        if copy_images:
            dest_dir = output_imagedir if split == "main" else output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def _make_createml_annotation_data(dic):
    """ makes createML annotations of a particular image"""

    category = dic["category"]
    del dic["category"]
    return {"label": category, "coordinates": dic}


def convert_createml(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """converts to createml .json from master dataframe

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
        save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
            annotation. Defaults to ``None``.
    """
    save_under = ifnone(save_under, "createml")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()
    for split in splits:
        # output_subdir = output_labeldir if split == "main" else output_labeldir / split
        # output_subdir.mkdir(parents=True, exist_ok=True)

        split_df = df.query("split == @split").copy()
        # drop images missing width or height information
        hw_missing = split_df[pd.isnull(split_df["image_width"]) | pd.isnull(split_df["image_height"])]
        if len(hw_missing) > 0:
            warnings.warn(
                f"{hw_missing['image_id'].nunique()} has height/width information missing in split `{split}`. "
                + f"{len(hw_missing)} annotations will be removed."
            )

        split_df = split_df[pd.notnull(split_df["image_width"]) & pd.notnull(split_df["image_height"])]
        split_df = split_df.rename(columns={"y_min": "y", "x_min": "x"})
        grouped_split_df = split_df.groupby(["image_id"])

        createml_data = []
        for image_info, grouped_info in tqdm(grouped_split_df, total=grouped_split_df.ngroups, desc=f"split: {split}"):
            file_result = {}
            records = grouped_info[["category", "height", "width", "y", "x"]].to_dict("records")
            file_result["image"] = image_info
            # transform the records into createml annotation format
            file_result["annotations"] = list(map(_make_createml_annotation_data, records))
            createml_data.append(file_result)

        file_path = output_labeldir / f"{split}.json"
        write_json(createml_data, file_path)

        if copy_images:
            dest_dir = output_imagedir if split == "main" else output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def convert_pascal(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
):
    """convert to pascal from Masterdf

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional):  If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional):  Whether to copy the images to a different directory. Defaults to False.
        save_under (Optional[str], optional):  Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional):  Output directory for the target
    """

    save_under = ifnone(save_under, "pascal")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    df = copy.deepcopy(df)
    df["x_max"] = df["x_min"] + df["width"]
    df["y_max"] = df["y_min"] + df["height"]
    df.drop(["width", "height"], axis=1, inplace=True)

    for col in ("x_min", "y_min", "x_max", "y_max"):
        df[col] = df[col].astype(np.int32)
    splits = df.split.unique().tolist()

    for split in splits:
        output_subdir = output_labeldir / split if len(splits) > 1 else output_labeldir
        output_subdir.mkdir(parents=True, exist_ok=True)
        split_df = df.query("split == @split")
        images = split_df["image_id"].unique()

        for image in images:
            image_df = split_df.query("image_id == @image")
            write_xml(image_df, root, output_labeldir)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def convert_tfrecord(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    has_image_split: bool = False,
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """convert to tfrecords  from Masterdf

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional):  If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional):  Whether to copy the images to a different directory. Defaults to False.
        save_under (Optional[str], optional):  Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional):  Output directory for the target
    """
    import tensorflow as tf

    output_dir = ifnone(output_dir, root, Path)
    save_under = ifnone(save_under, "tfrecord")
    output_dir = output_dir / save_under
    output_imagedir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = copy.deepcopy(df)
    df["x_max"] = df["x_min"] + df["width"]
    df["y_max"] = df["y_min"] + df["height"]
    df.drop(["width", "height"], axis=1, inplace=True)

    for col in ("x_min", "y_min", "x_max", "y_max"):
        df[col] = df[col].astype(np.int32)
    splits = df.split.unique().tolist()
    for split in splits:
        split_df = df.query("split == @split")
        writer = tf.io.TFRecordWriter(str(Path(output_dir).joinpath(split + ".tfrecord")))
        images = split_df["image_id"].unique()
        for image in images:
            image_df = split_df.query("image_id == @image")
            tf_example = create_tf_example(image_df, root)
            writer.write(tf_example.SerializeToString())
        writer.close()
    id_to_class_map = get_id_to_class_map(df)
    write_label_map(id_to_class_map, output_dir)

    if copy_images:
        dest_dir = output_imagedir / split
        dest_dir.mkdir(parents=True, exist_ok=True)

        _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def convert_simple_json(
    df: pd.DataFrame,
    root: Union[str, os.PathLike, PosixPath],
    copy_images: bool = False,
    save_under: Optional[str] = None,
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """converts to simple json from master df

    Args:
        df (pd.DataFrame): the master df
        root (Union[str, os.PathLike, PosixPath]): root directory of the source format
        has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
        copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
        save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
        output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
            annotation. Defaults to ``None``.
    """

    save_under = ifnone(save_under, "simple_json")
    output_imagedir, output_labeldir = _makedirs(root, save_under, output_dir)

    splits = df.split.unique().tolist()

    for split in splits:
        split_df = df.query("split == @split").copy()
        split_df_columns = split_df.columns.to_list()
        is_score = True if "score" in split_df_columns else False
        images = split_df["image_id"].unique().tolist()

        simple_json_dict = {}
        image_groups = split_df.groupby("image_id")
        for image in tqdm(images, desc=f"split: {split}"):
            image_anns = image_groups.get_group(image)
            ann_cols = ["x_min", "y_min", "width", "height", "category"]
            if is_score:
                ann_cols.append("score")
            image_anns = image_anns[ann_cols].to_dict("records")
            simple_json_dict[image] = _create_simple_json_dict(image_anns)

        output_file = output_labeldir / f"{split}.json"
        write_json(simple_json_dict, output_file)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            _fastcopy(split_df["image_path"].unique().tolist(), dest_dir)


def _create_simple_json_dict(image_anns):
    """Makes a list of annotations in simple_json format."""
    simple_json_anns = []
    for ann in image_anns:
        ann_dict = {}
        ann_dict["bbox"] = [ann["x_min"], ann["y_min"], ann["width"] + ann["x_min"], ann["height"] + ann["y_min"]]
        ann_dict["classname"] = ann["category"]
        if "score" in ann.keys():
            ann_dict["confidence"] = ann["score"]
        simple_json_anns.append(ann_dict)
    return simple_json_anns
