import random
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from optical import Annotation
from optical.converter import FormatSpec
from optical.converter.converter import (  # noqa: F401
    convert_coco,
    convert_createml,
    convert_csv,
    convert_pascal,
    convert_sagemaker,
    convert_simple_json,
    convert_tfrecord,
    convert_yolo,
)
from optical.converter.utils import find_splits

EXTS = {
    "coco": "json",
    "yolo": "txt",
    "pascal": "xml",
    "createml": "json",
    "sagemaker": "manifest",
    "csv": "csv",
    "simple_json": "json",
}


@pytest.mark.parametrize("format", ["coco", "createml", "csv", "sagemaker", "yolo", "pascal"])
@pytest.mark.parametrize("has_split", [False, True])
def test_split(dirs):
    images_dir = Path(dirs[0]) / "images"
    annotation_dir = Path(dirs[0]) / "annotations"
    splits, has_im_split = find_splits(images_dir, annotation_dir, dirs[1])

    assert set(splits) == set(dirs[2])
    assert has_im_split == dirs[3]


@pytest.mark.parametrize("fmt", ["coco", "createml", "sagemaker", "yolo", "pascal", "csv", "tfrecord", "simple_json"])
@pytest.mark.parametrize("has_split", [False, True])
def test_resolve_df(root, fmt, has_split):

    annotation = Annotation(root, format=fmt)
    assert annotation.format == fmt
    assert len(annotation.label_df) == 439

    if has_split:
        assert annotation.label_df.query("split == 'train'").shape[0] == 316
        assert annotation.label_df.query("split == 'valid'").shape[0] == 123

    for _ in range(10):
        row = random.randint(0, len(annotation.label_df) - 1)
        assert Path(annotation.label_df.image_path.iloc[row]).is_file()


@pytest.mark.parametrize("fmt", ["coco", "createml", "sagemaker", "yolo", "pascal", "csv", "simple_json"])
def test_export(test_dir, label_df, fmt):
    exporter = eval(f"convert_{fmt}")
    export_dir = Path(test_dir).joinpath("exports")
    exporter(df=label_df, root=export_dir)

    if fmt in ("coco", "createml", "csv", "sagemaker", "simple_json"):
        assert export_dir.joinpath(fmt).joinpath("annotations").joinpath(f"train.{EXTS[fmt]}").is_file()
        assert export_dir.joinpath(fmt).joinpath("annotations").joinpath(f"valid.{EXTS[fmt]}").is_file()
    else:
        assert export_dir.joinpath(fmt).joinpath("annotations").joinpath("train").is_dir()
        assert export_dir.joinpath(fmt).joinpath("annotations").joinpath("valid").is_dir()


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [False])
@pytest.mark.parametrize("stratified", [False, True])
def test_split_dataset(root, fmt, stratified):
    annotation = Annotation(root, format=fmt)
    splits = annotation.train_test_split(test_size=0.2, stratified=stratified)

    assert splits.master_df.query("split == 'train'").shape[0] > 0
    assert splits.master_df.query("split == 'valid'").shape[0] > 0


def test_formatspec_attrs(test_dir, label_df):
    fmtspec = FormatSpec(Path(test_dir).joinpath("testfiles"), has_split=True, df=label_df)

    desc = fmtspec.describe()
    bb_stats = fmtspec.bbox_stats()
    dist = fmtspec.show_distribution()
    scatter = fmtspec.bbox_scatter()

    assert isinstance(desc, pd.DataFrame)
    assert isinstance(bb_stats, pd.DataFrame)
    assert isinstance(dist, alt.Chart)
    assert isinstance(scatter, alt.Chart)


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
def test_tfrecord_export(root, fmt, test_dir):
    annotation = Annotation(root, format=fmt)

    output_dir = Path(test_dir).joinpath("exports")
    annotation.export(to="tfrecord", output_dir=output_dir)

    assert output_dir.joinpath("tfrecord").joinpath("train.tfrecord").is_file()
    assert output_dir.joinpath("tfrecord").joinpath("valid.tfrecord").is_file()
