"""
__author__: HashTagML
license: MIT
Created: Monday, 29th March 2021
"""

import argparse
import copy
import json
from pathlib import Path, PosixPath
import os
import shutil
import warnings
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

from tqdm.auto import tqdm

PathType = Union[str, os.PathLike, PosixPath]
COCODict = List[Dict[str, Union[int, str, List[int]]]]

Label = namedtuple("Label", ["class_index", "bbox"])


class CocoToYolo:
    def __init__(self, only_annon):
        self._category_ref = {}
        self.only_annon = only_annon
        self._num_images = 0
        self._classes = None

    def _get_coco_cats(self, coco_json: COCODict) -> Dict[int, str]:
        """return coco categories"""
        categories = coco_json["categories"]

        for cat_idx, cat in enumerate(categories):
            if not cat["id"] in self._category_ref:
                self._category_ref[cat["id"]] = cat_idx

        print(self._category_ref)
        return dict(
            zip(
                [self._category_ref[cat["id"]] for cat in categories],
                [cat["name"] for cat in categories],
            ),
        )

    def _parse_images(self, coco_json: COCODict) -> Dict[int, Any]:
        """extracts image metadata from COCO json and returns a dict with keys as ``id`` of the images

        Args:
            coco_json (COCODict): input COCO json

        Returns:
            Dict[int, Any]: image metadata dict
        """
        images = copy.deepcopy(coco_json["images"])
        im_dict = dict()
        for _, img in enumerate(images):
            im_id = img.pop("id")
            im_dict[im_id] = img
        return im_dict

    def _bbox_coco_to_yolo(self, bbox: List[int], img_dims: Tuple[int, int]) -> List[float]:
        """transforms COCO coordinates to YOLO coordinates with normalisation.

        Args:
            bbox (List[int]): list of coordinates ``(x_min, y_min, width, height)``
            img_dims (Tuple[int, int]): dimension of the image ``(width, height)``

        Returns:
            List[float]: the normalised YOLO coordinates
        """
        x, y, width, height = bbox
        w, h = img_dims
        c_x, c_y = x + width / 2.0, y + height / 2.0
        return [c_x / w, c_y / h, width / w, height / h]

    def coco2yolo(self, coco_json: COCODict):
        """parses a COCO json and retuns the image metadata and label information. The image metadata the is a dict
            with ``image_id`` as the key and the metadata as values. The labels are namedtuple of the format
            :math:``(class_index, [x_{center}, y_{center}, {width}, {height}])``

        Args:
            coco_json (COCODict): a COCO JSON object

        Returns:
            image metadata and label information.
        """

        _ = self._get_coco_cats(coco_json)
        im_dict = self._parse_images(coco_json)

        annotations = coco_json["annotations"]
        labels = dict()
        for _, ann in enumerate(annotations):
            key = ann["image_id"]
            class_idx = self._category_ref[ann["category_id"]]
            bbox = ann["bbox"]

            w, h = im_dict[key]["width"], im_dict[key]["height"]
            bbox = self._bbox_coco_to_yolo(bbox, (w, h))
            val = Label(class_idx, bbox)
            if key in labels:
                labels[key].append(val)
            else:
                labels[key] = [val]
        return im_dict, labels

    def _get_yolo_string(self, label):
        return str(label.class_index) + " " + " ".join([str(x) for x in label.bbox])

    def save_yolo(
        self,
        coco_json: PathType,
        output_imgs_dir: PathType,
        output_labels_dir: PathType,
        coco_images_dir: PathType,
        print_warn: bool = False,
    ):
        Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
        im_dict, yolo_labels = self.coco2yolo(coco_json)
        nim, na = 0, 0
        for im in tqdm(im_dict, total=len(im_dict)):
            try:
                labels = yolo_labels[im]
                txt_file = Path(output_labels_dir).joinpath(im_dict[im]["file_name"].split(".")[0] + ".txt")
                with open(txt_file, "w") as f:
                    for label in labels:
                        line = self._get_yolo_string(label)
                        f.write(line)
                        f.write("\n")
            except KeyError:
                na += 1
                if print_warn:
                    warnings.warn(f"no annotation found for {im_dict[im]['file_name']}, skipping...")

            if not self.only_annon:
                try:
                    img_name = im_dict[im]["file_name"]
                    img_src_path = Path(coco_images_dir) / img_name
                    img_dst_path = Path(output_imgs_dir) / img_name
                    shutil.copy(img_src_path, img_dst_path)

                except FileNotFoundError:
                    nim += 1
                    if print_warn:
                        warnings.warn(f"Image not found {im_dict[im]['file_name']} while moving, skipping...")
        return nim, na

    def transform_coco_to_yolo(
        self,
        json_path: PathType,
        output_imgs_dir: PathType,
        output_labels_dir: PathType,
        coco_images_dir: PathType,
        print_warn: bool = False,
    ):
        with open(json_path, "r") as f:
            coco_json = json.load(f)
        # print(f"transforming {Path(json_path).name} and saving results to {output_dir}...")
        num_im_missing, num_annon_missing = self.save_yolo(
            coco_json, output_imgs_dir, output_labels_dir, coco_images_dir, print_warn
        )
        if num_im_missing > 0:
            print(f"{num_im_missing} images could not be found while conversion.")
        if num_annon_missing > 0:
            print(f"{num_annon_missing} annotations could not be found while conversion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform annotations from COCO to YOLO format")
    parser.add_argument("-c", "--coco_dir", dest="coco_dir", help="input directory")
    parser.add_argument("-o", "--output_dir", dest="output_dir", help="output directory")
    parser.add_argument(
        "-comp",
        "--compress",
        dest="compress",
        action="store_true",
        help="whether to compress labels into zip format.",
    )
    parser.add_argument(
        "-oann",
        "--only_annon",
        action="store_true",
        help="whether to convert only labels",
    )
    parser.add_argument(
        "-p",
        "--print-warn",
        dest="print_warn",
        action="store_true",
        help="whether to print warnings for missing annotations and images warnings.",
    )
    args = parser.parse_args()

    annotations = (
        Path(args.coco_dir).joinpath("annotations").glob("*.json")
    )  # glob.glob(os.path.join(args.coco_dir, "annotations", "*.json"))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    transform_obj = CocoToYolo(args.only_annon)
    coco_images_dir = Path(args.coco_dir) / "images"
    for ann in annotations:
        output_subdir = Path(ann).name.split(".")[0]
        output_imgs_subdir = (
            Path(output_dir) / "images" / output_subdir
        )  # os.path.join(output_dir, "images", output_subdir)
        output_labels_subdir = (
            Path(output_dir) / "labels" / output_subdir
        )  # os.path.join(output_dir, "labels", output_subdir)

        Path(output_labels_subdir).mkdir(
            parents=True, exist_ok=True
        )  # os.makedirs(output_labels_subdir, exist_ok=True)
        if not args.only_annon:
            Path(output_imgs_subdir).mkdir(parents=True, exist_ok=True)

        transform_obj.transform_coco_to_yolo(
            ann,
            output_imgs_subdir,
            output_labels_subdir,
            coco_images_dir,
            args.print_warn,
        )
        if args.compress:
            shutil.make_archive(output_labels_subdir, "zip", output_labels_subdir)
