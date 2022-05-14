"""
__author__: HashTagML
license: MIT
Created: Tuesday, 30th March 2021
"""
# TODO: needs better solution for Handling TFrecords

import os
from pathlib import Path
from typing import Optional, Union

import altair as alt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

from .export import get_exporter
from .utils import CopyType, DetectionFormat, Pathlike, find_splits, ifnone, partiion_df

pd.options.mode.chained_assignment = None


class FormatSpec:
    """The base class to represent all annotation formats"""

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        has_split: Optional[bool] = False,
        df: Optional[pd.DataFrame] = None,
        format: Optional[str] = None,
    ):
        self.root = Path(root)
        self._has_image_split = has_split
        self.master_df = df
        self._format = format
        self._splits = None

    # @abstractmethod
    # removing absract class as it cannot be instantiated from within
    # as required for split
    def _resolve_dataframe(self):
        pass

    def __str__(self):
        return f"{self.format.upper()}[root:{self.root}, splits:[{', '.join(self.splits)}]]"

    def __repr__(self):
        return self.format

    @property
    def format(self):
        if self._format is None:
            return self.__module__.split(".")[-1]
        return self._format

    @property
    def splits(self):
        return self._splits

    def _find_splits(self):
        splits, has_image_split = find_splits(self._image_dir, self._annotation_dir, self.format)
        self._has_image_split = has_image_split
        self._splits = splits

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None) -> pd.DataFrame:
        """computes bbox descriptive stats e.g., mean, std etc.

        Args:
            split (Optional[str]): split of the dataset e.g., ``train``, ``valid`` etc. Defaults to None.
            category (Optional[str]): category to filter out. Defaults to None.

        Returns:
            pd.DataFrame: stats of the bounding boxes
        """
        df = partiion_df(self.master_df, split, category)
        return df[["x_min", "y_min", "width", "height"]].describe()

    def show_distribution(self) -> alt.Chart:
        """Plots distribution of labels in different splits of the dataset"""

        df = self.master_df[["split", "category", "image_id"]].copy()
        distribution = df.groupby(["split", "category"])["image_id"].size().rename("count")
        distribution = pd.DataFrame(distribution / distribution.groupby(level=0).sum()).reset_index()

        return (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x="category:O", y="count:Q", color="category", column="split")
        )

    def bbox_scatter(
        self, split: Optional[str] = None, category: Optional[str] = None, limit: int = 1000
    ) -> alt.Chart:
        """plots scatter of width and height of bounding boxes

        Args:
            split (Optional[str]): split of the dataset e.g., ``train``, ``valid`` etc. Defaults to None.
            category (Optional[str]): category to filter out. Defaults to None.
            limit (int, optional): number of samples to plot. Defaults to 1000.
        """

        df = partiion_df(self.master_df, split, category).drop("image_path", axis=1)
        limit = min(min(limit, len(df)), 5000)
        df = df.sample(n=limit, replace=False, random_state=42)
        return alt.Chart(df).mark_circle(size=30).encode(x="width", y="height", color="category")

    def describe(self) -> pd.DataFrame:
        """shows basic data distribution in different split"""
        df = (
            self.master_df.groupby(["split"])
            .agg({"image_id": [pd.Series.nunique, "size"], "category": pd.Series.nunique})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + "_" + df.columns.get_level_values(1)
        df.rename(
            columns={
                "image_id_nunique": "images",
                "image_id_size": "annotations",
                "category_nunique": "categories",
                "split_": "split",
            },
            inplace=True,
        )
        return df

    def split(self, test_size: float = 0.2, stratified: bool = False, random_state: int = 42):
        """splits the dataset into train and validation sets

        Args:
            test_size (float, optional): Fraction of total images to be kept for validation. Defaults to 0.2.
            stratified (bool, optional): Whether to stratify the data splits. Defaults to False.
            random_state (int, optional): Random state for the split. Defaults to 42.

        Returns:
            FormatSpec: Returns an instance of `FormatSpec` class
        """

        label_df = self.master_df.copy()

        if stratified:
            class_df = label_df[["image_id", "class_id"]].copy()
            class_df.drop_duplicates(inplace=True)
            gdf = class_df.groupby("image_id")["class_id"].agg(lambda x: x.tolist()).reset_index()

            mlb = MultiLabelBinarizer()
            out = mlb.fit_transform(gdf.class_id)
            label_names = [f"class_{x}" for x in mlb.classes_]
            out = pd.DataFrame(data=out, columns=label_names)

            gdf = pd.concat([gdf, out], axis=1)
            gdf.drop(["class_id"], axis=1, inplace=True)

            train_images, _, test_images, _ = iterative_train_test_split(
                gdf[["image_id"]].values, gdf[label_names].values, test_size=test_size
            )

            train_images = train_images.ravel()
            test_images = test_images.ravel()

        else:
            image_ids = label_df.image_id.unique()
            train_images, test_images = train_test_split(image_ids, test_size=test_size, random_state=random_state)

        train_df = label_df.loc[label_df["image_id"].isin(train_images.tolist())]
        test_df = label_df.loc[label_df["image_id"].isin(test_images.tolist())]

        train_df.loc[:, "split"] = "train"
        test_df.loc[:, "split"] = "valid"

        master_df = pd.concat([train_df, test_df], ignore_index=True)
        return FormatSpec(self.root, True, master_df, format=self.format)

    def save(
        self, output_dir: Optional[Union[str, os.PathLike]], export_to: Optional[str] = None, copy_images: bool = True
    ):
        """Just another api for convert. Similar to export"""
        export_to = ifnone(export_to, self.format)
        return self.convert(export_to, output_dir=output_dir, copy_images=copy_images)

    def convert(
        self,
        to: str,
        output_dir: Optional[Pathlike] = None,
        prefix: Optional[str] = None,
        copy_images: Optional[CopyType] = None,
        **kwargs,
    ):
        dest_format = DetectionFormat(to)
        exporter = get_exporter(dest_format)
        exporter.export(
            df=self.master_df, root=self.root, output_dir=output_dir, prefix=prefix, copy_images=copy_images
        )
