"""
__author__: HashTagML
license: MIT
Created: Thursday, 8th April 2021
"""
import math
import os
import warnings
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype

from .utils import (
    Resizer,
    check_df_cols,
    check_num_imgs,
    check_save_path,
    get_class_color_map,
    plot_boxes,
    render_grid_mpl,
    render_grid_mpy,
    render_grid_pil,
)


class Visualizer:
    """Creates visualizer to visualze images with annotations by batch size, name and index.
    Required dataframe of the dataset as input.Can show all images with annotations as a video.

    Args:
        images_dir (Union[str, os.PathLike]): Path to images in the dataset
        dataframe (pd.DataFrame): Pandas dataframe which is created by ``optical.converter``. Must contain
            ``["image_id", "x_min", "y_min", "width", "height", "category", "class_id"]`` columns.
        split (Optional[str], optional): Split of the dataset to be visualized.
        img_size (int, optional): Image size to resize and maintain uniformity. Defaults to 512.


    """

    def __init__(
        self,
        images_dir: Union[str, os.PathLike],
        dataframe: pd.DataFrame,
        split: Optional[str] = None,
        img_size: int = 512,
    ):
        # Check images dir and dataframe
        assert check_num_imgs(images_dir), f"No images found in {(images_dir)}, Please check."
        req_cols = ["image_id", "x_min", "y_min", "width", "height", "category", "class_id"]
        assert check_df_cols(
            dataframe.columns.to_list(), req_cols=req_cols
        ), f"Some required columns are not present in the dataframe.\
        Columns required for visualizing the annotations are {','.join(req_cols)}."

        # Initialization
        self.images_dir = images_dir
        self.resize = (img_size, img_size)
        self.original_df = dataframe
        if split is not None:
            self.original_df = dataframe.query("split == @split").copy()
        self.filtered_df = self.original_df.copy()
        self.last_sequence = 0

        # Initialize class map and color class map.
        self.class_map = pd.Series(
            self.original_df.class_id.values.astype(int), index=self.original_df.category
        ).to_dict()
        self.class_map = {v: k for k, v in self.class_map.items()}
        self.class_color_map = get_class_color_map(self.class_map)
        self.previous_batch = []
        self.previous_args = {}

    def __getitem__(self, image_id: str, use_original: bool = False) -> Dict:
        """Fetches images and annotations from the input dataframe.

        Args:
            image_id (str): Image Id to be fetched.
            use_original (bool, optional): Whether to search image in original or filtered dataframe.
                Defaults to False.

        Returns:
            Dict: Python dictionary containing queried image and annotation information.
        """
        if use_original:
            img_df = self.original_df[self.original_df["image_id"] == image_id]
        else:
            img_df = self.filtered_df[self.filtered_df["image_id"] == image_id]
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

        image_path = os.path.join(self.images_dir, image_id)
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
    ) -> List[Dict]:
        """Fetches batch of images and annotations, applies filters if provided.

        Args:
            num_imgs (int, optional): Number of images and annotation to be fetched.
                if it is ``-1`` all images and annotations in the dataset will be returned.
                Defaults to 1.
            index (Optional[int], optional): Index of the image to be fetched. Defaults to None.
            name (Optional[str], optional): Name of the image to be fetched. Defaults to None.
            random (bool, optional): If ``True`` randomly selects ``num_imgs`` images otherwise follows a sequence.
                Defaults to True.
            do_filter (bool, optional): Wether to apply filtering or not. Defaults to False.

        Returns:
            List[Dict]: List of images and annotations info.
        """
        self.filtered_df = self._apply_filters(**kwargs) if do_filter else self.filtered_df
        unique_images = list(self.filtered_df.image_id.unique())
        use_original = kwargs.get("use_original", False)
        batch_img_indices = []

        if num_imgs == -1:
            batch_img_indices = list(self.original_df.image_id.unique()) if use_original else unique_images

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
        render: str = "pil",
        random: bool = True,
        **kwargs,
    ) -> Any:
        """Displays a batch of images based on input size.

        Args:
            num_imgs (int, optional): Number of images and their annotation to be visualized. Defaults to 9.
            previous (bool, optional): If ``True`` just displays last batch. Defaults to False.
            save_path (Optional[str], optional): Output path if images and annotations to be saved. Defaults to None.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and annotations.
                If ``pil``, uses ``Pillow`` to display the images and annotations.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".
            random (bool, optional): If ``True`` randomly selects ``num_imgs`` images otherwise follows a sequence.
                Defaults to True.


        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """
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

    def reset_filters(self):
        """Resets all the filters applied on original dataframe."""
        self.filtered_df = self.original_df.copy()

    def _render_image_grid(
        self,
        num_imgs: int,
        drawn_imgs: List,
        image_names: List[str],
        render: str = "pil",
        save_path: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Renders image and annotation grid based on given backend.

        Args:
            num_imgs (int): Number of images in the grid.
            drawn_imgs (List): List of images with annotations.
            image_names (List[str]): List of image names in the grid.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and annotations.
                If ``pil``, uses ``Pillow`` to display the images and annotations.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".
            save_path (Optional[str], optional): Output path if images and annotations to be saved. Defaults to None.

        Raises:
            RuntimeError: Raised if invalid rendering backend is given.

        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """

        cols = 2 if num_imgs <= 6 else 3
        cols = 1 if num_imgs == 1 else cols
        rows = math.ceil(num_imgs / cols)
        if render.lower() == "mpl":
            render_grid_mpl(drawn_imgs, image_names, num_imgs, cols, rows, self.resize[0], save_path, **kwargs)
        elif render.lower() == "pil":
            return render_grid_pil(drawn_imgs, image_names, num_imgs, cols, rows, self.resize[0], save_path, **kwargs)
        elif render.lower() == "mpy":
            return render_grid_mpy(drawn_imgs, image_names, **kwargs)
        else:
            raise RuntimeError("Invalid Image grid rendering format, should be either mpl or pil.")

    def _draw_images(self, batch: List[Dict], **kwargs) -> Tuple[List, List]:
        """Draws annotations on the images.

        Args:
            batch (List[Dict]): List of images and annotations info.

        Returns:
            Tuple[List, List]: Tuple of drawn images and their respective image names.
        """
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

    def _apply_filters(self, **kwargs) -> pd.DataFrame:
        """Applies filters on the original dataframe.

        Keyword Args:
            only_without_labels(bool): To filter images which do not have any annotations.
            only_with_labels(bool): To filter only images which have annotations.
            filter_categories(Union[str,List[]]): To filter annotations with given categories.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
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
                curr_df = curr_df[curr_df["category"].str.lower().isin(labels)]
        return curr_df

    def show_image(
        self,
        index: int = 0,
        name: Optional[str] = None,
        save_path: Optional[str] = None,
        render: str = "mpl",
        **kwargs,
    ) -> Any:
        """Displays images with annotation given index or name.

        Args:
            index (int, optional): Index of the image to be fetched. Defaults to 0.
            name (Optional[str], optional): Name of the image to be fetched. Defaults to None.
            save_path (Optional[str], optional): Output path if images and annotations to be saved. Defaults to None.
            render (str, optional): Rendering to be used. Available options are ``mpl``,``pil``,``mpy``.
                If ``mpl``, uses ``matplotlib`` to display the images and annotations.
                If ``pil``, uses ``Pillow`` to display the images and annotations.
                If ``mpy``, uses ``mediapy`` to display as video
                Defaults to "pil".

        Returns:
            Any: Incase of Pillow or mediapy rendering IPython media object will be returned.
        """
        if name is not None:
            batch = self._get_batch(index=None, name=name, **kwargs)
        else:
            batch = self._get_batch(index=index, name=None, **kwargs)

        drawn_img, image_name = self._draw_images(batch, **kwargs)
        if save_path is not None:
            save_path = check_save_path(save_path, image_name[0])
            drawn_img[0].save(save_path)

        if len(drawn_img) > 0:
            return self._render_image_grid(1, drawn_img, image_name, render, **kwargs)
        else:
            warnings.warn("No valid images found to visualize.")
            return

    def show_video(self, use_original: bool = True, **kwargs) -> Any:
        """Displays whole dataset as a video.

        Args:
            use_original(bool): Whether to original dataset or filtered dataset.Defaults to ``True``

        Keyword Args:
            show_image_name(bool): Whether to show image names in the video or not.
            image_time(float): How many seconds each should be displayed in the video.
                e.g: ``image_time = 1`` means each image will be displayed for one second.
                ``image_time = 0.5`` means each image will be displayed for half a second.

        Returns:
            Any: Returns IPython media object.
        """

        batch = self._get_batch(num_imgs=-1, use_original=use_original, **kwargs)
        drawn_imgs, image_names = self._draw_images(batch, **kwargs)

        if len(drawn_imgs) > 0:
            return self._render_image_grid(len(drawn_imgs), drawn_imgs, image_names, render="mpy", **kwargs)
        else:
            warnings.warn("No valid images found to visualize.")
            return
