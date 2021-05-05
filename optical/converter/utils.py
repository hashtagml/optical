"""
__author__: HashTagML
license: MIT
Created: Sunday, 28th March 2021
"""

import json
import io
import os
import shutil
import warnings
from pathlib import Path, PosixPath
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
from lxml import etree as xml
from PIL import Image
import xml.etree.ElementTree as ET

_TF_INSTALLED = True
try:
    import tensorflow as tf
except ImportError:
    _TF_INSTALLED = False


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
            pass
    return out


def exists(path: Union[str, os.PathLike]):
    """checks for whether a directory or file exists in the specified path"""
    if Path(path).is_dir():
        return "dir"

    if Path(path).is_file():
        return "file"

    return


def get_image_dir(root: Union[str, os.PathLike]):
    """returns image directory given a root directory"""
    return Path(root) / "images"


def get_annotation_dir(root: Union[str, os.PathLike]):
    """returns annotation directory given a root directory"""
    return Path(root) / "annotations"


def find_job_metadata_key(json_data: Dict):
    """finds metadata key for sagemaker manifest format"""
    for key in json_data.keys():
        if key.split("-")[-1] == "metadata":
            return key


def read_coco(coco_json: Union[str, os.PathLike]):
    """read a coco json and returns the images, annotations and categories dict separately"""
    with open(coco_json, "r") as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"], coco["categories"]


def write_json(data_dict: Dict, filename: Union[str, os.PathLike]):
    """writes json to disk"""
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=2)


def filter_split_category(
    df: pd.DataFrame, split: Optional[str] = None, category: Optional[str] = None
) -> pd.DataFrame:
    """given the label df, filters the dataframe by split and/or label category

    Args:
        df (pd.DataFrame): the label dataframe.
        split (Optional[str], optional): the dataset split e.g., ``train``, ``test`` etc. Defaults to None.
        category (Optional[str], optional): the label category. Defaults to None.

    Raises:
        ValueError: if an unknown category is specified.

    Returns:
        pd.DataFrame: the filtered dataframe.
    """
    if split is not None:
        df = df.query("split == @split").copy()

    if category is not None:
        if category not in df.category.unique():
            raise ValueError(f"class `{category}` is not present in annotations")
        df = df.query("category == @category").copy()

    return df


def copyfile(
    src: Union[str, os.PathLike], dest: Union[str, os.PathLike], filename: Optional[Union[str, os.PathLike]] = None
) -> None:
    """copies a file from one path to another

    Args:
        src (Union[str, os.PathLike]): either a directory containing files or any filepath.
        dest (Union[str, os.PathLike]): the output directory for the copy.
        filename (Optional[Union[str, os.PathLike]], optional): If ``src`` is a directory, the name of the
           file to copy. Defaults to None.
    """
    if filename is not None:
        filename = Path(src) / filename

    else:
        filename = Path(src)

    dest = Path(dest) / filename.name
    try:
        shutil.copyfile(filename, dest)
    except FileNotFoundError:
        pass


def write_xml(
    df: pd.DataFrame,
    image_root: Union[str, os.PathLike, PosixPath],
    output_dir: Optional[Union[str, os.PathLike, PosixPath]] = None,
) -> None:
    """write xml files in PASCAL VOC format given a label dataframe

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


def get_id_to_class_map(df: pd.DataFrame):
    """This function return the class_id to class name mapping

    Args:
        df (pd.DataFrame): master dataframe

    Returns:
        Dict: mapping dictionary
    """
    set_df = df.drop_duplicates(subset="class_id")[["category", "class_id"]]
    return set_df.set_index("class_id")["category"].to_dict()


def find_splits(image_dir: Union[str, os.PathLike], annotation_dir: Union[str, os.PathLike], format: str):
    """find the splits in the dataset, will ignore splits for which no annotation is found"""

    # print(f"passed format: {format}")

    exts = {
        "coco": "json",
        "csv": "csv",
        "pascal": "xml",
        "yolo": "txt",
        "sagemaker": "manifest",
        "createml": "json",
        "simple_json": "json",
    }

    ext = exts[format]

    im_splits = [x.name for x in Path(image_dir).iterdir() if x.is_dir() and not x.name.startswith(".")]

    if format in ("yolo", "pascal"):
        ann_splits = [x.name for x in Path(annotation_dir).iterdir() if x.is_dir()]

        if not ann_splits:
            files = list(Path(annotation_dir).glob(f"*.{ext}"))
            if len(files):
                ann_splits = ["main"]
            else:
                raise ValueError("No annotation found. Please check the directory specified.")

    else:
        ann_splits = [x.stem for x in Path(annotation_dir).glob(f"*.{ext}")]

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


def _tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tf_float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _tf_bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _tf_int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(df: pd.DataFrame, root: Union[str, os.PathLike, PosixPath]):
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
                "image/height": _tf_int64_feature(height),
                "image/width": _tf_int64_feature(width),
                "image/filename": _tf_bytes_feature(filename),
                "image/source_id": _tf_bytes_feature(filename),
                "image/encoded": _tf_bytes_feature(encoded_jpg),
                "image/format": _tf_bytes_feature(image_format),
                "image/object/bbox/xmin": _tf_float_list_feature(xmins),
                "image/object/bbox/xmax": _tf_float_list_feature(xmaxs),
                "image/object/bbox/ymin": _tf_float_list_feature(ymins),
                "image/object/bbox/ymax": _tf_float_list_feature(ymaxs),
                "image/object/class/text": _tf_bytes_list_feature(classes_text),
                "image/object/class/label": _tf_int64_list_feature(classes),
            }
        )
    )
    return tf_example


def write_label_map(id_to_class_map: Dict, output_dir: Union[str, os.PathLike, PosixPath]):
    """writes label_map used in tf object detection

    Args:
        id_to_class_map (Dict): mapping dictionary
        output_dir ([type]): output path
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


def tf_decode_image(root: Union[str, os.PathLike, PosixPath], data, split: Union[str, os.PathLike, PosixPath]):
    """Decodes images and save in images folder under root

    Args:
        root (Union[str, os.PathLike, PosixPath]): path to root directory
        data (tf.train.Example): single image example
        split (Union[str, os.PathLike, PosixPath]): split directory
    """
    img_filename = data["image/filename"].numpy().decode("utf-8")
    img = data["image/encoded"].numpy()
    im = Image.open(io.BytesIO(img))
    im.save(str(Path(root) / "images" / split / img_filename))


def read_xml(xml_folder: Union[str, os.PathLike, PosixPath], img_path: Union[str, os.PathLike, PosixPath]):
    """read xml files in the folder and return list's of information used to construct master_df

    Args:
        xml_folder (Union[str, os.PathLike, PosixPath]): Xml file folder
        img_path (Union[str, os.PathLike, PosixPath]): Image Directory
    """
    img_filenames = []
    img_widths = []
    img_heights = []
    cls_names = []
    x_mins = []
    y_mins = []
    box_widths = []
    box_heights = []
    img_paths = []
    xml_files = [x for x in Path(xml_folder).glob("*.xml")]
    for fxml in xml_files:
        tree = ET.parse(fxml)
        root = tree.getroot()
        img_filename = root.find("filename").text
        img_width = root.find("size").find("width").text
        img_height = root.find("size").find("height").text
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            x_min = int(obj.find("bndbox").find("xmin").text)
            y_min = int(obj.find("bndbox").find("ymin").text)
            box_width = int(obj.find("bndbox").find("xmax").text) - int(x_min)
            box_height = int(obj.find("bndbox").find("ymax").text) - int(y_min)
            img_filenames.append(img_filename)
            img_widths.append(img_width)
            img_heights.append(img_height)
            cls_names.append(cls_name)
            x_mins.append(x_min)
            y_mins.append(y_min)
            box_widths.append(box_width)
            box_heights.append(box_height)
            img_paths.append(str(img_path.joinpath(img_filename)))
    return img_filenames, img_widths, img_heights, cls_names, x_mins, y_mins, box_widths, box_heights, img_paths
