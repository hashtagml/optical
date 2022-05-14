"""
__author__: HashTagML
license: MIT
Created: Friday, 13th May 2022
"""

import copy
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import yaml
from lxml import etree as xml
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from .utils import (
    CopyType,
    DetectionFormat,
    Pathlike,
    copy_files,
    get_class_id_category_mapping,
    ifnone,
    makedirs,
    write_json,
)

try:
    import tensorflow as tf
except ImportError:
    pass


class Exporter(Protocol):
    def export(self, *args, **kwargs):
        """exports to a specified format"""


def get_exporter(format: DetectionFormat) -> Exporter:
    if format == DetectionFormat.COCO:
        return CocoExporter()
    if format == DetectionFormat.CREATEML:
        return CreateMLExporter()
    if format == DetectionFormat.CSV:
        return CSVExporter()
    if format == DetectionFormat.JSON:
        return JsonExporter()
    if format == DetectionFormat.PASCAL_VOC:
        return PascalVOCExporter()
    if format == DetectionFormat.SAGEMAKER_MANIFEST:
        return SagemakerManifestExporter()
    if format == DetectionFormat.TFRECORD:
        return TFRecordExporter()
    if format == DetectionFormat.YOLO:
        return YoloExporter()


class CocoExporter:
    """exports annotation dataframe to COCO format"""

    def _make_coco_images(self, df: pd.DataFrame, image_map: Dict) -> List:
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

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ) -> None:
        """converts to coco from master df

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): root directory of the source format
            copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
            prefix (str, optional): Name of the folder to save the target annotations. Defaults to ``labels``.
            output_dir (Optional[Pathlike], optional): Output directory for the target
                annotation. Defaults to ``None``.
        """

        prefix = ifnone(prefix, "coco")

        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

        splits = df.split.unique().tolist()

        for split in splits:
            split_df = df.query("split == @split").copy()
            images = df["image_id"].unique().tolist()

            image_map = dict(zip(images, range(len(images))))
            image_list = self._make_coco_images(split_df, image_map)
            annotation_list = self._make_coco_annotations(split_df, image_map)
            category_list = self._make_coco_categories(split_df)

            coco_dict = dict()
            coco_dict["images"] = image_list
            coco_dict["annotations"] = annotation_list
            coco_dict["categories"] = category_list

            output_file = output_labeldir / f"{split}.json"
            write_json(coco_dict, output_file)

            if copy_images:
                dest_dir = output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)

                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))


class CreateMLExporter:
    @staticmethod
    def _make_createml_annotation_data(d: Dict, /):
        """crates CreateML annotations of a particular image"""

        category = d["category"]
        del d["category"]
        return {"label": category, "coordinates": d}

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ):
        """exports annotation dataframe to CreateML format.

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): the root directory of source annotations
            copy_images (Optional[CopyType]): Whether to copy the images to target directory
            prefix (Optional[str], optional): Directory name for the target annotations. Defaults to ``createml``.
            output_dir (Optional[Pathlike], optional): Output directory. Defaults to None.
        """
        prefix = ifnone(prefix, "createml")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

        splits = df.split.unique().tolist()
        for split in splits:
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
            for image_info, grouped_info in tqdm(
                grouped_split_df, total=grouped_split_df.ngroups, desc=f"split: {split}"
            ):
                file_result = {}
                records = grouped_info[["category", "height", "width", "y", "x"]].to_dict("records")
                file_result["image"] = image_info
                # transform the records into createml annotation format
                file_result["annotations"] = list(map(self._make_createml_annotation_data, records))
                createml_data.append(file_result)

            file_path = output_labeldir / f"{split}.json"
            write_json(createml_data, file_path)

            if copy_images:
                dest_dir = output_imagedir if split == "main" else output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)
                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))


class CSVExporter:
    """exports annotation dataframe to CSV format."""

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ):
        """exports annotation dataframe to CSV format.

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): the root directory of source annotations
            copy_images (Optional[CopyType]): Whether to copy the images to target directory
            prefix (Optional[str], optional): Directory name for the target annotations. Defaults to ``csv``.
            output_dir (Optional[Pathlike], optional): Output directory. Defaults to None.
        """
        prefix = ifnone(prefix, "csv")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

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

                copy_files(image_paths, dest_dir, type=CopyType(copy_images))


class JsonExporter:
    def _create_simple_json_dict(self, image_anns: List[Dict]) -> List[Dict]:
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

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ) -> None:

        """exports annotation dataframe to Json format.

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): the root directory of source annotations
            copy_images (Optional[CopyType]): Whether to copy the images to target directory
            prefix (Optional[str], optional): Directory name for the target annotations. Defaults to ``json``.
            output_dir (Optional[Pathlike], optional): Output directory. Defaults to None.
        """

        prefix = ifnone(prefix, "json")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

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
                simple_json_dict[image] = self._create_simple_json_dict(image_anns)

            output_file = output_labeldir / f"{split}.json"
            write_json(simple_json_dict, output_file)

            if copy_images:
                dest_dir = output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)

                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))


class PascalVOCExporter:
    """exports annotation dataframe to Pascal VOC format."""

    def _write_xml(
        self,
        df: pd.DataFrame,
        image_root: Pathlike,
        output_dir: Optional[Pathlike] = None,
    ) -> None:
        """write xml files in Pascal VOC format given a label dataframe
        Args:
            df (pd.DataFrame): dataframe of the single image with multiple objects in it.
            image_root (Union[str, os.PathLike, PosixPath]): path to image directory.
            output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): output directory
        """
        root = xml.Element("annotation")
        folder = xml.Element("folder")
        folder.text = ""
        root.append(folder)
        filename = xml.Element("filename")
        filename.text = df.iloc[0]["image_id"]
        root.append(filename)
        path = xml.Element("path")
        path.text = str(Path(image_root) / "images" / df.iloc[0]["split"] / df.iloc[0]["image_id"])
        root.append(path)
        source = xml.Element("source")
        root.append(source)
        database = xml.Element("database")
        database.text = "UNKNOWN"
        source.append(database)
        size = xml.Element("size")
        root.append(size)
        width = xml.Element("width")
        width.text = str(df.iloc[0]["image_width"])
        size.append(width)
        height = xml.Element("height")
        height.text = str(df.iloc[0]["image_height"])
        size.append(height)
        depth = xml.Element("depth")
        depth.text = "3"
        size.append(depth)
        segmented = xml.Element("segmented")
        segmented.text = "0"
        root.append(segmented)

        for _, objec in df.iterrows():
            obj = xml.Element("object")
            root.append(obj)
            name = xml.Element("name")
            name.text = objec["category"]
            obj.append(name)
            pose = xml.Element("pose")
            pose.text = "Unspecified"
            obj.append(pose)
            truncated = xml.Element("truncated")
            truncated.text = "0"
            obj.append(truncated)
            difficult = xml.Element("difficult")
            difficult.text = "0"
            obj.append(difficult)
            occluded = xml.Element("occluded")
            occluded.text = "0"
            obj.append(occluded)
            bndbox = xml.Element("bndbox")
            obj.append(bndbox)
            xmin = xml.Element("xmin")
            xmin.text = str(objec["x_min"])
            bndbox.append(xmin)
            xmax = xml.Element("xmax")
            xmax.text = str(objec["x_max"])
            bndbox.append(xmax)
            ymin = xml.Element("ymin")
            ymin.text = str(objec["y_min"])
            bndbox.append(ymin)
            ymax = xml.Element("ymax")
            ymax.text = str(objec["y_max"])
            bndbox.append(ymax)
        tree = xml.ElementTree(root)
        f_name = Path(output_dir).joinpath(df.iloc[0]["split"], Path(df.iloc[0]["image_id"]).stem + ".xml")

        with open(f_name, "wb") as files:
            tree.write(files, pretty_print=True)

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ):

        """exports annotation dataframe to Pascal VOC format.

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): the root directory of source annotations
            copy_images (Optional[CopyType]): Whether to copy the images to target directory
            prefix (Optional[str], optional): Directory name for the target annotations. Defaults to ``pascal``.
            output_dir (Optional[Pathlike], optional): Output directory. Defaults to None.
        """

        prefix = ifnone(prefix, "pascal")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

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
                self._write_xml(image_df, root, output_labeldir)

            if copy_images:
                dest_dir = output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)

                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))


class SagemakerManifestExporter:
    """exports annotation dataframe to Sagemaker manifest format."""

    @staticmethod
    def _make_manifest_data(image_info: List, grouped_info: pd.DataFrame, job_name: str, id_to_class_map: Dict):
        # creating json like data for each row of df
        manifest_dict = dict()
        manifest_dict["source-ref"] = image_info[0]
        manifest_dict[f"{job_name}"] = {
            "image_size": [{"width": int(image_info[1]), "depth": 3, "height": int(image_info[2])}],
        }
        manifest_dict[f"{job_name}-metadata"] = {
            "job-name": job_name,
            "class-map": id_to_class_map,
            "creation-date": str(datetime.now()),
            "type": "groundtruth/object-detection",
        }
        annotations = grouped_info[["class_id", "height", "width", "top", "left"]].to_dict("records")
        # append annotations
        manifest_dict[f"{job_name}"]["annotations"] = annotations

        return manifest_dict

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
        job_name: str = "optical",
    ):
        """converts to sagemaker manifest from master dataframe

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): root directory of the source format
            has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
            copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
            prefix (str, optional): Name of the folder to save the target annotations. Defaults to ``sagemaker``.
            output_dir (Optional[Pathlike]], optional): Output directory
                annotation. Defaults to ``None``.
            job_name(Optional[str]): manifest job name for the output file. Defaults to optical
        """

        prefix = ifnone(prefix, "sagemaker")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

        splits = df.split.unique().tolist()
        for split in splits:
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
            id_to_class_map = get_class_id_category_mapping(split_df)
            grouped_split_df = split_df.groupby(["image_id", "image_width", "image_height"])

            with open(output_labeldir / f"{split}.manifest", "w") as f:
                for image_info, grouped_info in tqdm(
                    grouped_split_df, total=grouped_split_df.ngroups, desc=f"split: {split}"
                ):
                    manifest_dic = self._make_manifest_data(image_info, grouped_info, job_name, id_to_class_map)
                    f.write(json.dumps(manifest_dic) + "\n")

            if copy_images:
                dest_dir = output_imagedir if split == "main" else output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)
                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))


class YoloExporter:
    """exports annotation dataframe to Yolo format"""

    @staticmethod
    def _write_yolo_txt(filename: str, output_dir: Pathlike, yolo_string: str):
        filepath = Path(output_dir).joinpath(Path(filename).stem + ".txt")
        with open(filepath, "a") as f:
            f.write(yolo_string)
            f.write("\n")

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ):
        """converts to yolo from master dataframe

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Union[str, os.PathLike, PosixPath]): root directory of the source format
            has_image_split (bool, optional): If the images are arranged under the splits. Defaults to False.
            copy_images (bool, optional): Whether to copy the images to a different directory. Defaults to False.
            save_under (str, optional): Name of the folder to save the target annotations. Defaults to "labels".
            output_dir (Optional[Union[str, os.PathLike, PosixPath]], optional): Output directory for the target
                annotation. Defaults to ``None``.
        """

        prefix = ifnone(prefix, "yolo")
        output_imagedir, output_labeldir = makedirs(root, prefix, output_dir)

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
                self._write_yolo_txt(image_id, output_subdir, ystr)

            if copy_images:
                dest_dir = output_imagedir / split
                dest_dir.mkdir(parents=True, exist_ok=True)

                copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))

        dataset["nc"] = len(lbl.classes_)
        dataset["names"] = list(lbl.classes_)

        with open(Path(output_labeldir).joinpath("dataset.yaml"), "w") as f:
            yaml.dump(dataset, f, default_flow_style=None, allow_unicode=True)


class TFRecordExporter:
    """exports annotation dataframe to TFRecord format."""

    def _write_label_map(id_to_class_map: Dict[int, str], output_dir: Pathlike):
        """writes label_map used in tf object detection

        Args:
            id_to_class_map (Dict[int, str]): mapping of class indices to class labels
            output_dir (Pathlike): output path
        """
        with open(output_dir.joinpath("label_map.pbtxt"), "w") as f:
            for id, cl in id_to_class_map.items():
                f.write("item\n")
                f.write("{\n")
                f.write("name :'{0}'".format(str(cl)))
                f.write("\n")
                f.write("id :{}".format(int(id)))
                f.write("\n")
                f.write("display_name:'{0}'".format(str(cl)))
                f.write("\n")
                f.write("}\n")

    def _tf_int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _tf_bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _tf_float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _tf_bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _tf_int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _create_tf_example(self, df: pd.DataFrame, root: Pathlike):
        """returns protobuf for a given image

        Args:
            df (pd.DataFrame): Dataframe of a single image with multiple records of objects
            root (Union[str, os.PathLike, PosixPath]): root of the Image path

        Returns:
            protobuf: protobuf of each Image
        """

        img_path = str(df["image_path"].iloc[0])
        with tf.io.gfile.GFile(img_path, "rb") as fid:
            encoded_jpg = fid.read()
        width = df.iloc[0]["image_width"]
        height = df.iloc[0]["image_height"]

        filename = df["image_id"].iloc[0].encode("utf8")
        image_format = b"jpg"
        xmins = list(df["x_min"] / width)
        xmaxs = list((df["x_max"]) / width)
        ymins = list(df["y_min"] / height)
        ymaxs = list((df["y_max"]) / height)
        classes_text = [s.encode("utf8") for s in df["category"]]
        classes = list(df["class_id"].astype(int))

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": self._tf_int64_feature(height),
                    "image/width": self._tf_int64_feature(width),
                    "image/filename": self._tf_bytes_feature(filename),
                    "image/source_id": self._tf_bytes_feature(filename),
                    "image/encoded": self._tf_bytes_feature(encoded_jpg),
                    "image/format": self._tf_bytes_feature(image_format),
                    "image/object/bbox/xmin": self._tf_float_list_feature(xmins),
                    "image/object/bbox/xmax": self._tf_float_list_feature(xmaxs),
                    "image/object/bbox/ymin": self._tf_float_list_feature(ymins),
                    "image/object/bbox/ymax": self._tf_float_list_feature(ymaxs),
                    "image/object/class/text": self._tf_bytes_list_feature(classes_text),
                    "image/object/class/label": self._tf_int64_list_feature(classes),
                }
            )
        )
        return tf_example

    def export(
        self,
        df: pd.DataFrame,
        root: Pathlike,
        copy_images: Optional[CopyType],
        prefix: Optional[str] = None,
        output_dir: Optional[Pathlike] = None,
    ) -> None:

        """exports annotation dataframe to TFRecord format.

        Args:
            df (pd.DataFrame): the annotation dataframe
            root (Pathlike): the root directory of source annotations
            copy_images (Optional[CopyType]): Whether to copy the images to target directory
            prefix (Optional[str], optional): Directory name for the target annotations. Defaults to ``tfrecord``.
            output_dir (Optional[Pathlike], optional): Output directory. Defaults to None.
        """
        output_dir = ifnone(output_dir, root, Path)
        prefix = ifnone(prefix, "tfrecord")
        output_dir = output_dir / prefix
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
                tf_example = self._create_tf_example(image_df, root)
                writer.write(tf_example.SerializeToString())
            writer.close()
        id_to_class_map = get_class_id_category_mapping(df)
        self._write_label_map(id_to_class_map, output_dir)

        if copy_images:
            dest_dir = output_imagedir / split
            dest_dir.mkdir(parents=True, exist_ok=True)

            copy_files(split_df["image_path"].unique().tolist(), dest_dir, type=CopyType(copy_images))
