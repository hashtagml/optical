"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

from .base import FormatSpec
from typing import Union
import os


class SageMaker(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        pass

    def _resolve_dataframe(self):
        return super()._resolve_dataframe()
