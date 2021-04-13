"""
__author__: HashTagML
license: MIT
Created: Tuesday, 30th March 2021
"""

from abc import ABC, abstractmethod
from typing import Optional

import altair as alt
import pandas as pd

from .converter import convert_coco, convert_csv, convert_yolo
from .utils import filter_split_category


class FormatSpec(ABC):
    """The base class to represent all annotation formats"""

    def __init__(self):
        self.root = None
        self._has_image_split = None
        self.master_df = None

    @abstractmethod
    def _resolve_dataframe(self):
        pass

    def bbox_stats(self, split: Optional[str] = None, category: Optional[str] = None) -> pd.DataFrame:
        """computes bbox descriptive stats e.g., mean, std etc.

        Args:
            split (Optional[str]): split of the dataset e.g., ``train``, ``valid`` etc. Defaults to None.
            category (Optional[str]): category to filter out. Defaults to None.

        Returns:
            pd.DataFrame: stats of the bounding boxes
        """
        df = filter_split_category(self.master_df, split, category)
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

        df = filter_split_category(self.master_df, split, category)
        limit = min(min(limit, len(df)), 5000)
        print(len(df))
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

    def convert(
        self, to: str, output_dir: Optional[str] = None, save_under: Optional[str] = None, copy_images: bool = False
    ):
        if to.lower() == "yolo":
            return convert_yolo(
                self.master_df,
                self.root,
                has_image_split=self._has_image_split,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        if to.lower() == "coco":
            return convert_coco(
                self.master_df,
                self.root,
                has_image_split=self._has_image_split,
                copy_images=copy_images,
                save_under=save_under,
                output_dir=output_dir,
            )
        if to.lower() == "pascal":
            pass
        if to.lower() == "csv":
            return convert_csv(
                self.master_df,
                self.root,
                has_image_split=self._has_image_split,
                output_dir=output_dir,
                save_under=save_under,
                copy_images=copy_images,
            )
        if to.lower() == "sagemaker":
            pass
