"""
__author__: HashTagML
license: MIT
Created: Thursday, 8th April 2021
"""
import math
import os
from random import shuffle
from typing import Optional, Union
import warnings

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
    render_grid_mpy,
    get_class_color_map,
    check_df_cols,
)


class Visualizer:
    def __init__(
        self,
        images_dir: Union[str, os.PathLike],
        dataframe: pd.DataFrame,
        split: Optional[str] = None,
        img_size: int = 512,
    ):
        assert check_num_imgs(images_dir), f"No images found in {(images_dir)}, Please check."
        req_cols = ["image_id", "x_min", "y_min", "width", "height", "category", "class_id"]
        assert check_df_cols(
            dataframe.columns.to_list(), req_cols=req_cols
        ), f"Some required columns are not present in the dataframe.\
        Columns required for visualizing the annotations are {','.join(req_cols)}."
        self.images_dir = images_dir
        self.resize = (img_size, img_size)
        self.original_df = dataframe
        if split is not None:
            self.original_df = dataframe.query("split == @split").copy()
        self.filtered_df = self.original_df.copy()
        self.last_sequence = 0
        self.class_map = pd.Series(
            self.original_df.class_id.values.astype(int), index=self.original_df.category
        ).to_dict()
        self.class_map = {v: k for k, v in self.class_map.items()}
        self.class_color_map = get_class_color_map(self.class_map)
        self.previous_batch = []
        self.previous_args = {}

    def __getitem__(self, index, use_original):
        if use_original:
            img_df = self.original_df[self.original_df["image_id"] == index]
        else:
            img_df = self.filtered_df[self.filtered_df["image_id"] == index]
        img_bboxes = []
        scores = []
        for _, row in img_df.iterrows():
            is_valid_row = is_numeric_dtype(
                pd.Series([row["x_min"], row["y_min"], row["width"], row["height"], row["class_id"]])
            )
            if is_valid_row:
                box = [row["x_min"], row["y_min"], row["x_min"] + row["width"], row["y_min"] + row["height"]]
                img_bboxes.append(box + [row["class_id"]])
                if "score" in row.keys():
                    scores.append(row["score"])

        image_path = os.path.join(self.images_dir, index)
        img, anns = Resizer(self.resize)({"image_path": image_path, "anns": img_bboxes})
        item = {"img": {"image_name": image_path, "image": img}, "anns": anns}
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

        self.filtered_df = self._apply_filters(**kwargs) if do_filter else self.filtered_df
        unique_images = list(self.filtered_df.image_id.unique())
        use_original = False
        batch_img_indices = []

        if num_imgs == -1:
            batch_img_indices = unique_images

        elif index is not None or name is not None:
            unique_images_original = list(self.original_df.image_id.unique())
            if index is not None:
                index = index % len(unique_images_original)
                batch_img_indices = [unique_images_original[index]]
                use_original = True
            elif name is not None and name in unique_images_original:
                batch_img_indices = [name]
                use_original = True
            else:
                print(f"{name} not found in the dataset. Please check")

        else:
            actual_num_images = min(len(unique_images), num_imgs)
            if actual_num_images < num_imgs:
                warnings.warn(f"Found only {actual_num_images} in the dataset.")
            if random:
                shuffle(unique_images)
                batch_img_indices = unique_images[:actual_num_images]
            else:
                start_index = self.last_sequence
                end_index = self.last_sequence + actual_num_images
                self.last_sequence = end_index if end_index <= len(unique_images) else 0
                batch_img_indices = unique_images[start_index:end_index]

        backend = "threading"
        r = Parallel(n_jobs=-1, backend=backend)(
            delayed(self.__getitem__)(idx, use_original) for idx in batch_img_indices
        )
        return r

    def show_batch(
        self,
        num_imgs: int = 9,
        previous: bool = False,
        save_path: Optional[str] = None,
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

        drawn_imgs, image_names = self._draw_images(batch, **kwargs)
        if num_imgs != len(drawn_imgs):
            num_imgs = len(drawn_imgs)
            warnings.warn(f"Visualizing only {num_imgs} images.")

        if num_imgs == 1:
            if save_path is not None:
                save_path = check_save_path(save_path)
                drawn_imgs[0].save(image_names[0] + "_vis.jpg")
            return drawn_imgs[0]

        if len(drawn_imgs) > 0:
            return self._render_image_grid(num_imgs, drawn_imgs, image_names, render, **kwargs)
        else:
            warnings.warn("No valid images found to visualize.")
            return

    def _render_image_grid(
        self,
        num_imgs,
        drawn_imgs,
        image_names,
        render: str = "mpl",
        save_path: Optional[str] = None,
        **kwargs,
    ):

        cols = 2 if num_imgs <= 6 else 3
        cols = 1 if num_imgs == 1 else cols
        rows = math.ceil(num_imgs / cols)
        if render.lower() == "mpl":
            render_grid_mpl(drawn_imgs, image_names, num_imgs, cols, rows, self.resize[0], save_path, **kwargs)
        elif render.lower() == "pil":
            return render_grid_pil(drawn_imgs, image_names, num_imgs, cols, rows, self.resize[0], save_path, **kwargs)
        elif render.lower() == "mpy":
            return render_grid_mpy(drawn_imgs, image_names, num_imgs, cols, rows, self.resize[0], save_path, **kwargs)
        else:
            raise RuntimeError("Invalid Image grid rendering format, should be either mpl or pil.")

    def _draw_images(self, batch, **kwargs):
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
                warnings.warn(f"Could not plot bounding boxes for {img_name}")
                continue

        return drawn_imgs, image_names

    def _apply_filters(self, **kwargs):
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
                        warnings.warn(f"{label} category is not present in the dataset. Please check")
            if len(labels) > 0:
                curr_df = curr_df[curr_df["category"].isin(labels)]
        return curr_df

    def show_image(
        self,
        index: int = 0,
        name: Optional[str] = None,
        save_path: Optional[str] = None,
        render: str = "mpl",
        **kwargs,
    ):
        if name is not None:
            batch = self._get_batch(index=None, name=name, **kwargs)
        else:
            batch = self._get_batch(index=index, name=None, **kwargs)

        drawn_img, image_name = self._draw_images(batch, **kwargs)
        if save_path is not None:
            save_path = check_save_path(save_path, image_name[0])
            drawn_img[0].save(save_path)

        if len(drawn_img) > 0:
            return self._render_image_grid(1, drawn_img, image_name, render)
        else:
            warnings.warn("No valid images found to visualize.")
            return

    def show_video(self, **kwargs):

        batch = self._get_batch(num_imgs=-1, **kwargs)
        drawn_imgs, image_names = self._draw_images(batch, **kwargs)

        if len(drawn_imgs) > 0:
            return self._render_image_grid(len(drawn_imgs), drawn_imgs, image_names, render="mpy")
        else:
            warnings.warn("No valid images found to visualize.")
            return
