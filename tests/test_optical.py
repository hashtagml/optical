import random
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from optical.detection import Annotation
from optical.detection.formats import FormatSpec
from optical.detection.formats.export import get_exporter
from optical.detection.formats.utils import DetectionFormat, find_splits

EXTS = {
    "coco": "json",
    "yolo": "txt",
    "pascal_voc": "xml",
    "createml": "json",
    "sagemaker_manifest": "manifest",
    "csv": "csv",
    "json": "json",
}


@pytest.mark.parametrize(
    "format",
    [
        DetectionFormat.COCO,
        DetectionFormat.CREATEML,
        DetectionFormat.CSV,
        DetectionFormat.PASCAL_VOC,
        DetectionFormat.SAGEMAKER_MANIFEST,
        DetectionFormat.YOLO,
    ],
)
@pytest.mark.parametrize("has_split", [False, True])
def test_split(dirs):
    images_dir = Path(dirs[0]) / "images"
    annotation_dir = Path(dirs[0]) / "annotations"
    splits, has_im_split = find_splits(images_dir, annotation_dir, dirs[1])

    assert set(splits) == set(dirs[2])
    assert has_im_split == dirs[3]


@pytest.mark.parametrize(
    "fmt", ["coco", "createml", "sagemaker_manifest", "yolo", "pascal_voc", "csv", "tfrecord", "json"]
)
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


@pytest.mark.parametrize(
    "fmt",
    [
        DetectionFormat.COCO,
        DetectionFormat.CREATEML,
        DetectionFormat.SAGEMAKER_MANIFEST,
        DetectionFormat.YOLO,
        DetectionFormat.PASCAL_VOC,
        DetectionFormat.CSV,
        DetectionFormat.JSON,
    ],
)
def test_export(test_dir, label_df, fmt):
    exporter = get_exporter(fmt)
    export_dir = Path(test_dir).joinpath("exports")
    exporter.export(df=label_df, root=export_dir)

    if fmt in (
        DetectionFormat.COCO,
        DetectionFormat.CREATEML,
        DetectionFormat.CSV,
        DetectionFormat.SAGEMAKER_MANIFEST,
        DetectionFormat.JSON,
    ):
        assert export_dir.joinpath(fmt.value).joinpath("annotations").joinpath(f"train.{EXTS[fmt]}").is_file()
        assert export_dir.joinpath(fmt.value).joinpath("annotations").joinpath(f"valid.{EXTS[fmt]}").is_file()
    else:
        assert export_dir.joinpath(fmt.value).joinpath("annotations").joinpath("train").is_dir()
        assert export_dir.joinpath(fmt.value).joinpath("annotations").joinpath("valid").is_dir()


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
