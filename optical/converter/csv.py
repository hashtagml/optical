"""
__author__: HashTagML
license: MIT
Created: Wednesday, 31st March 2021
"""

import os
from typing import Union

from .base import FormatSpec


class Csv(FormatSpec):
    def __init__(self, root: Union[str, os.PathLike]):
        pass

    def _resolve_dataframe(self):
        return super()._resolve_dataframe()
