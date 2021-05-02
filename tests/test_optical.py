from pathlib import Path

import pytest

from optical.converter.utils import find_splits
from optical import Annotation


@pytest.mark.parametrize("format", ["coco", "createml", "csv", "sagemaker", "yolo", "pascal"])
@pytest.mark.parametrize("has_split", [False, True])
def test_split(dirs):
    images_dir = Path(dirs[0]) / "images"
    annotation_dir = Path(dirs[0]) / "annotations"
    splits, has_im_split = find_splits(images_dir, annotation_dir, dirs[1])

    assert set(splits) == set(dirs[2])
    assert has_im_split == dirs[3]


@pytest.mark.parametrize("fmt", ["coco", "createml", "csv", "sagemaker", "yolo", "pascal"])
@pytest.mark.parametrize("has_split", [False, True])
def test_resolve_df(root):
    root_, frmt = root
    annotation = Annotation(root_, format=frmt)
    assert len(annotation.label_df) == 439
