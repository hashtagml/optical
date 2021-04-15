"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""
import imagesize
from pathlib import Path
import warnings
from base import FormatSpec
from typing import Union
import os
import numpy as np
import pandas as pd

from utils import exists, get_image_dir, get_annotation_dir
from num2words import num2words


class Yolo(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = root
        self._image_dir = get_image_dir(root)
        self._annotation_dir = get_annotation_dir(root)
        self._has_image_split = False
        assert exists(self._image_dir), "root is missing `images` directory."
        assert exists(self._annotation_dir), "root is missing `annotations` directory."
        self._splits = self._find_splits()
        self._resolve_dataframe()
        
    def _find_splits(self):
        im_splits = [x.name for x in Path(self._image_dir).iterdir() if x.is_dir()]
        ann_splits = [x.name for x in Path(self._annotation_dir).iterdir() if x.is_dir()]

        if im_splits:
            self._has_image_split = True
        no_anns = set(im_splits).difference(ann_splits)
        if no_anns:
            warnings.warn(f"no annotation found for {','.join(list(no_anns))}")
        return ann_splits
    
    def _resolve_dataframe(self):
        img_names = []
        cls_ids = []
        x_center = []
        y_center = []
        box_width = []
        box_height = []
        splits = []
        image_height= []
        image_width= []              
        for split in self._splits:
            ann_dir_files = self._annotation_dir/f"{split}"
            img_dir_files = self._image_dir/f"{split}"
            txt_files = [x for x in Path(ann_dir_files).glob("*.txt")]
            img_files = [x for x in Path(img_dir_files).glob("*.jpg")]
            for txt, img in zip(txt_files,img_files):
                file_names = os.path.basename(img)
                image_widths,image_heights = imagesize.get(img)
                image_height.append(image_heights)
                image_width.append(image_widths)
                with open(txt, 'rt') as fd:
                    first_line = fd.readline()
                    splited = first_line.split()
                    class_id, x_cent, y_cent, box_widths, box_heights = splited[0], splited[1], splited[2], splited[3], splited[4]
                    img_names.append(file_names)
                    x_center.append(x_cent)
                    y_center.append(y_cent)
                    box_width.append(box_widths)
                    box_height.append(box_heights)
                    cls_ids.append(class_id)
                    splits.append(split)
                    
        category = [num2words(i) for i in cls_ids]      
        master_df = pd.DataFrame(
            list(
                zip(
                    img_names,
                    image_width,
                    image_height,
                    cls_ids,
                    category,
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                    splits,
                 )
              ),
            columns = [
                 "image_id",
                 "image_width",
                 "image_height",
                 "class_id",
                 "category",
                 "x_min",
                 "y_min",
                 "width",
                 "height",
                 "split",
            ],
        )      
        
        if len(master_df[pd.isnull(master_df.image_id)]) > 0:
            warnings.warn(
                    "There are annotations in your dataset for which there is no matching images"
                    + f"(in split `{split}`). These annotations will be removed during any "
                    + "computation or conversion. It is recommended that you clean your dataset."
                )  
        for column in ["x_min","y_min","width","height"]:
            master_df[column] = master_df[column].astype(np.float32)     
        for column in ["image_width","image_height"]:
            master_df[column] = master_df[column].astype(np.int32)    
        for column in ["category"]:
            master_df[column] = master_df[column].astype(str)
        for column in ["class_id"]:
            master_df[column] = master_df[column].astype(np.int32)
            
        self.master_df = master_df
