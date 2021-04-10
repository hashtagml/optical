"""
__author__: HashTagML
license: MIT
Created: Thursday, 8th April 2021
"""
import math
import os
from random import shuffle
from typing import Optional, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from joblib import Parallel, delayed

from .utils import (
    Resizer,
    check_num_imgs,
    check_save_path,
    plot_boxes,
    render_grid_mpl,
    render_grid_pil,
    get_class_color_map,
)


class Visualizer:
    def __init__(
        self,
        images_dir: Union[str, os.PathLike],
        dataframe: pd.DataFrame,
        split: Optional[str] = None,
        img_size: int = 512,
    ):
        self.images_dir = images_dir
        assert check_num_imgs(self.images_dir), f"No images found in {(self.images_dir)}, Please check."
        self.resize = (img_size, img_size)
        self.original_df = dataframe
        if split is not None:
            self.original_df = dataframe.query("split == @split").copy()
        self.filtered_df = self.original_df.copy()
        self.last_sequence = 0
        self.class_map = pd.Series(self.original_df.class_id.values, index=self.master_df.category).to_dict()
        self.class_color_map = get_class_color_map(self.class_map)

    def __getitem__(self, index):
        img_df = self.filtered_df[self.filtered_df["image_path"] == index]
        img_bboxes = []
        scores = []
        for _, row in img_df.iterrows():
            is_valid_row = is_numeric_dtype(
                pd.Series([row["xmin"], row["ymin"], row["width"], row["height"], row["class_id"]])
            )
            if is_valid_row:
                box = [row["xmin"], row["ymin"], row["xmin"] + row["width"], row["ymin"] + row["height"]]
                img_bboxes.append(box + [row["class_id"]])
                if "score" in row.keys():
                    scores.append(row["score"])

        img, anns = Resizer(self.resize)({"image_path": index, "anns": img_bboxes})
        item = {"img": {"image_name": index.name, "image": img}, "anns": anns}
        if len(scores) > 0:
            item["scores"] = scores
        del img_df
        return item

    def _get_batch(
        self,
        num_imgs: int = 1,
        index: Optional[int] = None,
        name: Optional[str] = None,
        random: bool = True,
        do_filter: bool = False,
        **kwargs,
    ):

        self.filtered_df = self.apply_filters(**kwargs) if do_filter else self.filtered_df
        unique_images = list(self.filtered_df.image_path.unique())

        batch_img_indices = []
        if index is not None or name is not None:
            if index is not None:
                index = index % len(unique_images)
                batch_img_indices = unique_images[index]
            elif name is not None and name in unique_images:
                batch_img_indices = [name]
            else:
                print(f"{name} not found in the dataset. Please check")

        else:
            actual_num_images = min(len(unique_images), num_imgs)
            if actual_num_images < num_imgs:
                print(f"Found only {actual_num_images} in the dataset.")
            if random:
                shuffle(unique_images)
                batch_img_indices = unique_images[:actual_num_images]
            else:
                start_index = self.last_sequence
                end_index = self.last_sequence + actual_num_images
                self.last_sequence = end_index - 1 if end_index <= len(unique_images) else 0
                batch_img_indices = unique_images[start_index:end_index]

        backend = "threading"
        r = Parallel(n_jobs=-1, backend=backend)(delayed(self.__getitem__)(idx) for idx in batch_img_indices)
        return r

    def show_batch(
        self,
        num_imgs: Optional[int] = 9,
        previous: Optional[bool] = False,
        save_path: Optional[bool] = False,
        render: str = "PIL",
        random: bool = True,
        **kwargs,
    ):
        if previous and len(self.previous_batch):
            batch = self.previous_batch
        else:
            do_filter = True
            if kwargs == self.previous_args:
                do_filter = False
            batch = self._get_batch(num_imgs, random=random, do_filter=do_filter, **kwargs)
            self.previous_batch = batch
            self.previous_args = kwargs

        drawn_imgs, image_names = self.draw_images(batch, **kwargs)
        if num_imgs != len(drawn_imgs):
            num_imgs = len(drawn_imgs)
            print(f"Visualizing only {num_imgs} images.")

        if num_imgs == 1:
            if save_path is not None:
                save_path = check_save_path(save_path)
                drawn_imgs[0].save(image_names[0] + "_vis.jpg")
            return drawn_imgs[0]

        cols = 2 if num_imgs <= 6 else 3
        rows = math.ceil(num_imgs / cols)
        if render.lower() == "mpl":
            render_grid_mpl(
                drawn_imgs,
                image_names,
                num_imgs,
                cols,
                rows,
                self.resize,
                save_path,
            )
        elif render.lower() == "pil":
            return render_grid_pil(
                drawn_imgs,
                image_names,
                num_imgs,
                cols,
                rows,
                self.resize[0],
                save_path,
            )
        else:
            raise RuntimeError("Invalid Image grid rendering format, should be either mpl or pil.")

    def draw_images(self, batch, **kwargs):
        drawn_imgs = []
        image_names = []
        for img_ann_info in batch:
            img_name = img_ann_info["img"]["image_name"]
            img = img_ann_info["img"]["image"]
            anns = img_ann_info["anns"]
            scores = img_ann_info.get("scores", None)
            try:
                drawn_img = plot_boxes(
                    img, anns, scores, class_map=self.class_map, class_color_map=self.class_color_map, **kwargs
                )
                image_names.append(img_name)
                drawn_imgs.append(drawn_img)
            except Exception:
                print(f"Could not plot bounding boxes for {img_name}")
                continue

        return drawn_imgs, image_names

    def apply_filters(self, **kwargs):
        if kwargs.get("only_without_labels", None):
            df = self.original_df[self.df["class_id"].isna() & self.df["category"].isna()]
            return df
        curr_df = self.original_df.copy()
        if kwargs.get("only_with_labels", None):
            curr_df = self.original_df.dropna(axis=0)
        if kwargs.get("filter_categories", None):
            filter_labels = kwargs["filter_categories"]
            ds_classes = [cat.lower() for cat in list(self.original_df.category.unique())]
            labels = []
            if len(filter_labels) > 0:
                labels = [filter_labels] if isinstance(filter_labels, str) else filter_labels
                labels = [cat.lower() for cat in labels]
                for label in labels:
                    if label not in ds_classes:
                        print(f"{label} category is not present in the dataset. Please check")
            if len(labels) > 0:
                curr_df = curr_df[curr_df["category"].isin(labels)]
        return curr_df

    def show_image(self, index: int = 0, name: Optional[str] = None, save_path: Optional[str] = None, **kwargs):
        if name is not None:
            batch = self._get_batch(index=None, name=name, **kwargs)
        else:
            batch = self._get_batch(index=index, name=None, **kwargs)

        drawn_img, image_name = self.draw_images(batch, **kwargs)
        if save_path is not None:
            save_path = check_save_path(save_path, image_name)
            drawn_img[0].save(save_path)
        return drawn_img[0]
