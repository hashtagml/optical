"""
__author__: HashTagML
license: MIT
Created: Friday, 13th May 2022
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import ImageFont

_FONT_PATH = str(Path(__file__).absolute().parent.joinpath("NotoSansMono-Black.ttf"))


@dataclass
class ColorPalette:
    fore: Tuple[int, int, int]
    text: Tuple[int, int, int]
    rgb: bool = True
    background: Tuple[int, int, int] = field(init=False, default_factory=tuple)

    def __post_init__(self):
        self.background = self.fore

    def rgb_to_bgr(self):
        return ColorPalette(fore=self.fore[::-1], text=self.text[::-1], rgb=False)


class Color(ColorPalette, Enum):
    YELLOW = ((255, 216, 70), (103, 87, 28))
    BLUE = ((0, 120, 210), (173, 220, 252))
    ORANGE = ((255, 125, 57), (104, 48, 19))
    LIME = ((1, 255, 127), (0, 102, 53))
    RED = ((255, 47, 65), (131, 0, 17))
    GREEN = ((0, 204, 84), (15, 64, 31))
    MAROON = ((135, 13, 75), (239, 117, 173))
    NAVY = ((0, 38, 63), (119, 193, 250))
    FUCHSIA = ((246, 0, 184), (103, 0, 78))
    TEAL = ((15, 205, 202), (0, 0, 0))
    GRAY = ((168, 168, 168), (0, 0, 0))
    PURPLE = ((179, 17, 193), (241, 167, 244))
    OLIVE = ((52, 153, 114), (25, 58, 45))
    BLACK = ((24, 24, 24), (220, 220, 220))
    AQUA = ((115, 221, 252), (0, 76, 100))
    SILVER = ((220, 220, 220), (0, 0, 0))


class Shape(Protocol):
    """defines shapes for plotting on images"""

    def draw(self):
        """draw the shape on an image"""


@dataclass
class Box:
    """Bounding box object"""

    left: int
    top: int
    right: int
    bottom: int
    _cast_int: bool = True

    def __post_init__(self):
        assert self.right > self.left, "Invalid box coordinates"
        assert self.bottom > self.top, "Invalid box coordinates"
        if self._cast_int:
            self.left = int(self.left)
            self.top = int(self.top)
            self.right = int(self.right)
            self.bottom = int(self.bottom)

    def draw(self, image: NDArray, color: ColorPalette, thickness: int):
        """draw rectangle on image

        Args:
            image (np.ndarray): image
            color (list): color in RGB list
            thickness (int): thickness of the shape
        """
        cv2.rectangle(image, (self.left, self.top), (self.right, self.bottom), color.fore, thickness)

    def add_label(
        self,
        image: NDArray,
        label: str,
        color: ColorPalette,
    ):

        _, image_width, _ = image.shape

        label_image = self._get_label_image(label, color, 15)
        label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = self.top
        rectangle_left = max(0, min(self.left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = self.top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        cv2.rectangle(image, rec_left_top, rec_right_bottom, color.fore, -1)
        try:
            image[label_top:label_bottom, label_left:label_right, :] = label_image
        except ValueError:
            pass

    def _color_image(self, image: NDArray, font_color: int, background_color: int) -> NDArray:
        """return's color image  of label from binary(black and white) image channel wise

        Args:
            image (np.ndarray): black and white image
            font_color (int): label color of a particular channel
            background_color (int): background color of the particular channel

        Returns:
            NDArray: colored  label image of a particular channel
        """
        return background_color + (font_color - background_color) * image / 255

    def _get_label_image(self, text: str, color: ColorPalette, font_height: int) -> NDArray:
        """return's colored label image which is to be added to the main image

        Args:
            text (str): label
            color (ColorPalette): color definition for the bounding box
            font_height (int): height of the font

        Returns:
            NDArray: color label image
        """
        font = ImageFont.truetype(_FONT_PATH, font_height)
        text_image = font.getmask(text)
        shape = list(reversed(text_image.size))
        bw_image = np.array(text_image).reshape(shape)

        font_colors = color.text
        background_colors = color.background

        image = [
            self._color_image(bw_image, font_color, background_color)[None, ...]
            for font_color, background_color in zip(font_colors, background_colors)
        ]

        return np.concatenate(image).transpose(1, 2, 0)
