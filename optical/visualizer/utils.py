"""
__author__: HashTagML
license: MIT
Created: Thursday, 8th April 2021
"""
import collections
import copy
import os
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import bounding_box.bounding_box as bb
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageDraw, ImageOps

from .config import COLORS, IMAGE_BORDER, IMAGE_EXT


def check_num_imgs(images_dir: Union[str, os.PathLike]) -> int:
    """Checks number of images in given directory"""
    file_counts = collections.Counter(p.suffix for p in images_dir.iterdir())
    return sum([file_counts.get(ext, 0) for ext in IMAGE_EXT])


def check_df_cols(df_cols: List, req_cols: List) -> bool:
    """Verifies whether input dataframe contains required columns or not.

    Args:
        df_cols (List): List of columns in the input dataframe.
        req_cols (List): List of required columns.

    Returns:
        bool: ``True`` if all required columns are present, otherwise ``False``.
    """
    for r_col in req_cols:
        if r_col not in df_cols:
            return False
    return True


class Resizer(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, expected_size: Tuple = (512, 512)):
        assert isinstance(expected_size, tuple)
        self.expected_size = expected_size

    def __call__(self, sample):

        img_path, anns = sample["image_path"], sample["anns"]
        img = self._get_resized_img(img_path)
        bboxes = self._regress_boxes(anns)
        return img, bboxes

    def _set_letterbox_dims(self):
        """ Get letterbox resize dimensions of the images."""
        iw, ih = self.orig_dim
        ew, eh = self.expected_size

        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        self.new_dim = (nw, nh)

        offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2
        self.offset = (offset_x, offset_y)

        upsample_x, upsample_y = iw / nw, ih / nh
        self.upsample = (upsample_x, upsample_y)

    def _get_resized_img(self, img_path: str):
        """Resizes the image."""
        img = Image.open(img_path)
        self.orig_dim = img.size
        self._set_letterbox_dims()
        img = img.resize(self.new_dim)
        new_img = Image.new("RGB", self.expected_size, color=(255, 255, 255))
        new_img.paste(img, self.offset)
        return new_img

    def _regress_boxes(self, bboxes: np.ndarray):
        """Regress the bounding boxes based on resize."""

        if not len(bboxes):
            return []

        if not hasattr(bboxes, "ndim"):
            bboxes = np.array(bboxes)

        # bboxes[:, 2] += bboxes[:, 0]
        # bboxes[:, 3] += bboxes[:, 1]

        bboxes[:, 0] = bboxes[:, 0] / self.upsample[0]
        bboxes[:, 1] = bboxes[:, 1] / self.upsample[1]
        bboxes[:, 2] = bboxes[:, 2] / self.upsample[0]
        bboxes[:, 3] = bboxes[:, 3] / self.upsample[1]

        bboxes[:, 0] += self.offset[0]
        bboxes[:, 1] += self.offset[1]
        bboxes[:, 2] += self.offset[0]
        bboxes[:, 3] += self.offset[1]

        return bboxes


def plot_boxes(
    img: Image,
    bboxes: np.ndarray,
    scores: Optional[List] = None,
    class_map: Optional[Dict] = dict(),
    class_color_map: Optional[Dict] = dict(),
    **kwargs,
) -> Image:
    """Plots bounding boxes annotations on the images.

    Args:
        img (Image): Pillow image on which annotations to be drawn.
        bboxes (np.ndarray): Bounding boxes of the input image.
        scores (Optional[List], optional): Scores incase of simple json format. Defaults to None.
        class_map (Optional[Dict], optional): mapping between categories and their ids. Defaults to dict().
        class_color_map (Optional[Dict], optional): mapping between categories and their colors. Defaults to dict().

    Returns:
        Image: PIL images on which annotations are drawn.
    """
    draw_img = np.array(img)
    for i, box in enumerate(bboxes):
        bbox = list(map(lambda x: max(0, int(x)), box[:-1]))
        if not isinstance(box[-1], str):
            category = class_map.get(int(box[-1]), str(int(box[-1])))
        else:
            category = box[-1]
        if kwargs.get("truncate_label", None) is not None:
            category = "".join([cat[0].lower() for cat in category.split(kwargs.get("truncate_label"))])
        if scores is not None:
            category = category + ":" + str(round(scores[i], 2))
        color = class_color_map.get(int(box[-1]), "green")
        bb.add(draw_img, *bbox, category, color=color)
    return Image.fromarray(draw_img)


def check_save_path(save_path: Union[str, os.PathLike], name: str = None) -> str:
    """Validates output path, creates one if it does exist."""
    save_path = Path(save_path)
    Path.mkdir(save_path, parents=True, exist_ok=True)
    if save_path.suffix in IMAGE_EXT:
        return save_path
    else:
        file_name = "vis.jpg" if name is None else name
        return str(save_path.joinpath(file_name))


def get_class_color_map(class_map: Dict) -> Dict:
    """Maps categories in the dataset with random colors."""
    class_color_map = dict()
    avail_colors = copy.deepcopy(COLORS)
    for cat_id, _ in class_map.items():
        if len(avail_colors):
            color = random.choice(avail_colors)
        else:
            color = "green"
        class_color_map[cat_id] = color
        if color in avail_colors:
            avail_colors.remove(color)
    return class_color_map


def render_grid_mpl(
    drawn_imgs: List,
    image_names: List,
    num_imgs: int,
    cols: int,
    rows: int,
    img_size: int,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Uses ``matplotlib`` to render image grid.

    Args:
        drawn_imgs (List): List of images with annotations.
        image_names (List): List of image names to be drawn.
        num_imgs (int): Number of images in the batch
        cols (int): Number of columns required in the grid.
        rows (int): Number of rows required in the grid.
        img_size (int): Each resized image size.
        save_path (Optional[str], optional): Output path if images and annotations to be saved. Defaults to None.
    """
    fig = plt.figure(
        figsize=(
            (rows * img_size + 3 * IMAGE_BORDER * rows) / 72,
            (cols * img_size + 3 * IMAGE_BORDER * cols) / 72,
        )
    )
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(rows, cols),
        axes_pad=0.5,  # pad between axes in inch
    )
    for ax, im, im_name in zip(grid, drawn_imgs, image_names):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        im_name = im_name.split("/")[-1]
        title = "\n".join(textwrap.wrap(im_name, width=32))
        ax.set_title(title)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in grid[num_imgs:]:
        ax.axis("off")
    if save_path is not None:
        save_path = check_save_path(save_path)
        plt.savefig(save_path)
    plt.show()


def render_grid_mpy(drawn_imgs: List, image_names: List, **kwargs) -> Any:
    """Renders batch of images as a video.

    Args:
        drawn_imgs (List): List of images with annotations.
        image_names (List): List of image names to be drawn.

    Returns:
        Any: IPython media object.
    """
    if kwargs.get("show_image_name", None):
        drawn_imgs = [np.array(add_name_strip(img, name)) for img, name in zip(drawn_imgs, image_names)]
    else:
        drawn_imgs = [np.array(img) for img in drawn_imgs]
    fps = 1 if kwargs.get("image_time", None) is None else 1 / kwargs.get("image_time", 1.0)
    return media.show_video(drawn_imgs, fps=fps)


def add_name_strip(img: np.ndarray, name: str):
    """ Adds name to image at the top."""
    drawn_img = ImageOps.expand(img, border=IMAGE_BORDER, fill=(255, 255, 255))
    name = name.split("/")[-1]
    lines = textwrap.wrap(name, width=32)
    y_text = IMAGE_BORDER // 2 if len(lines) <= 1 else 0
    dimg = ImageDraw.Draw(drawn_img)
    font = dimg.getfont()
    w = drawn_img.size[0]
    for line in lines:
        width, height = font.getsize(line)
        dimg.multiline_text(((w - width) // 2, y_text), line, font=font, fill=(0, 0, 0))
        y_text += height
    return drawn_img


def render_grid_pil(
    drawn_imgs: List,
    image_names: List,
    num_imgs: int,
    cols: int,
    rows: int,
    img_size: int,
    save_path: Optional[str] = None,
    **kwargs,
) -> Any:
    """Uses ``matplotlib`` to render image grid.

    Args:
        drawn_imgs (List): List of images with annotations.
        image_names (List): List of image names to be drawn.
        num_imgs (int): Number of images in the batch
        cols (int): Number of columns required in the grid.
        rows (int): Number of rows required in the grid.
        img_size (int): Each resized image size.
        save_path (Optional[str], optional): Output path if images and annotations to be saved. Defaults to None.
    Returns:
        Any: IPython media object.
    """
    for i in range(len(drawn_imgs)):
        drawn_img = drawn_imgs[i]
        img_name = image_names[i]
        drawn_img = add_name_strip(drawn_img, img_name)
        drawn_imgs[i] = drawn_img

    width = cols * (img_size + 2 * IMAGE_BORDER)
    height = rows * (img_size + 2 * IMAGE_BORDER)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    idx = 0
    for y in range(0, height, img_size + 2 * IMAGE_BORDER + 1):
        for x in range(0, width, img_size + 2 * IMAGE_BORDER + 1):
            if idx < num_imgs:
                canvas.paste(drawn_imgs[idx], (x, y))
                idx += 1
    if save_path is not None:
        save_path = check_save_path(save_path)
        plt.savefig(save_path)
    return canvas
