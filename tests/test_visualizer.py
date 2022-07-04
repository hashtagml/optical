import unittest.mock as mock
from pathlib import Path
from shutil import rmtree

import IPython
import pytest
from optical import Annotation
from PIL import Image

EXTS = {
    "coco": "json",
    "yolo": "txt",
    "pascal": "xml",
    "createml": "json",
    "sagemaker": "manifest",
    "csv": "csv",
    "simple_json": "json",
}


@pytest.mark.parametrize("fmt", ["coco", "createml", "sagemaker", "yolo", "pascal", "csv", "tfrecord", "simple_json"])
@pytest.mark.parametrize("has_split", [False, True])
@pytest.mark.parametrize("split", ["train", "valid", None])
@pytest.mark.parametrize("img_size", [256])
@pytest.mark.parametrize("backend", ["pil"])
def test_pil_visualization(root, fmt, split, has_split, img_size, backend):

    annotation = Annotation(root, format=fmt)

    if split is None and has_split:
        with pytest.warns(UserWarning, match="Split not specified, *"):
            vis = annotation.visualizer(img_size=img_size)

    elif split is not None and has_split:
        vis = annotation.visualizer(split=split, img_size=img_size)

    else:
        vis = annotation.visualizer(img_size=img_size)

    mpl_block = False if backend == "mpl" else None
    img = vis.show_batch(num_imgs=3, render=backend, mpl_block=mpl_block)
    assert isinstance(img, Image.Image)


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
@pytest.mark.parametrize("img_size", [512])
@pytest.mark.parametrize("backend", ["mpl"])
def test_mpl_visualization(root, fmt, has_split, img_size, backend):

    split = "valid"
    annotation = Annotation(root, format=fmt)
    vis = annotation.visualizer(split=split, img_size=img_size)
    with mock.patch("matplotlib.pyplot.show") as mockmethod:
        vis.show_batch(num_imgs=3, render=backend)
    mockmethod.assert_called_once_with()


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
@pytest.mark.parametrize("img_size", [512])
@pytest.mark.parametrize("backend", ["mpy"])
def test_mpy_visualization(root, fmt, has_split, img_size, backend):

    split = "valid"
    annotation = Annotation(root, format=fmt)
    vis = annotation.visualizer(split=split, img_size=img_size)
    htmls = []
    with mock.patch("IPython.display.display", htmls.append):
        vis.show_batch(num_imgs=3, render=backend)
    assert len(htmls) == 1
    assert isinstance(htmls[0], IPython.display.HTML)


@pytest.mark.parametrize("fmt", ["simple_json"])
@pytest.mark.parametrize("has_split", [True])
def test_threshold_visualization(root, fmt, has_split, backend="pil"):

    split = "valid"
    annotation = Annotation(root, format=fmt)
    annotation.formatspec.master_df.loc[:, "score"] = 0.5

    with pytest.raises(AssertionError, match="Threshold shoule be between [0.,1.]*"):
        vis = annotation.visualizer(split=split, threshold=1.5)

    with pytest.warns(UserWarning, match=f"There are no images to be visualized in {split}."):
        vis = annotation.visualizer(split=split, threshold=0.6)

    vis = annotation.visualizer(split=split, threshold=0.3)
    vis.show_batch(num_imgs=3, render=backend, random=False)
    assert vis.last_sequence == 3


@pytest.mark.parametrize("fmt", ["simple_json"])
@pytest.mark.parametrize("has_split", [True])
def test_show_all(root, fmt, has_split, backend="pil"):

    split = "train"
    annotation = Annotation(root, format=fmt)
    unique_images = list(annotation.formatspec.master_df.image_id.unique())
    num_images = 3
    selected_images = unique_images[:num_images]
    only_select = annotation.formatspec.master_df.image_id.isin(selected_images)
    annotation.formatspec.master_df = annotation.formatspec.master_df[only_select]
    vis = annotation.visualizer(split=split)

    with pytest.warns(UserWarning, match=f"Visualizing only {num_images}*"):
        vis.show_batch(num_imgs=9)
        vis.show_batch(previous=True, render=backend)

    htmls = []
    with mock.patch("IPython.display.display", htmls.append):
        vis.show_video(show_image_name=True, image_time=1)
    assert len(htmls) == 1
    assert isinstance(htmls[0], IPython.display.HTML)


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
def test_visualization_errors(root, fmt):
    annotation = Annotation(root, format=fmt)

    annotation.formatspec.master_df.drop(["width"], axis=1, inplace=True)
    with pytest.raises(AssertionError, match="Some required columns are not present in the dataframe."):
        _ = annotation.visualizer(split="valid")

    annotation = Annotation(root, format=fmt)
    with pytest.raises(AssertionError, match="No images found in *"):
        _ = annotation.visualizer(image_dir=f"{root}/annotations", split="valid")

    annotation = Annotation(root, format=fmt)
    with pytest.raises(RuntimeError, match="Invalid Image grid rendering format,*"):
        vis = annotation.visualizer(split="valid")
        _ = vis.show_batch(num_imgs=3, render="pilll")

    annotation = Annotation(root, format=fmt)
    annotation.formatspec.master_df.loc[:, "x_min"] = 1e16
    annotation.formatspec.master_df.loc[:, "y_min"] = 1e16
    vis = annotation.visualizer(split="valid")
    with pytest.warns(UserWarning, match="Could not plot bounding boxes for*"):
        _ = vis.show_batch(num_imgs=1)

    with pytest.warns(UserWarning, match="No valid images found to visualize*"):
        _ = vis.show_batch(num_imgs=1)

    with pytest.warns(UserWarning, match="No valid images found to visualize*"):
        _ = vis.show_image(index=0, render="pil")

    with pytest.warns(UserWarning, match="No valid images found to visualize*"):
        _ = vis.show_video()


@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
def test_visualization_filters(root, fmt):
    annotation = Annotation(root, format=fmt)
    vis = annotation.visualizer(split="valid")

    # Test filtering when visualizing images without any labels.
    with pytest.warns(UserWarning, match="Could not find any valid images to visualize,"):
        vis.show_batch(num_imgs=9, only_without_labels=True)

    # Test filtering when visualizing images only with labels.
    _ = vis.show_batch(num_imgs=9, only_with_labels=True)

    # Test filtering when visualizing images with specific labels.
    filter_classes = ["car", "window"]
    _ = vis.show_batch(num_imgs=9, filter_categories=filter_classes)
    filtered_classes = sorted(list(vis.filtered_df.category.unique()))

    assert filter_classes == filtered_classes

    vis.reset_filters()

    # Test filtering with invalid class
    filter_classes = ["random_cls"]
    with pytest.warns(UserWarning, match=f"{filter_classes[0]} category is not present*"):
        _ = vis.show_batch(num_imgs=9, filter_categories=filter_classes)


@pytest.mark.parametrize("valid_name_idx", [(True, True, None), (False, True, None), (False, False, 6)])
@pytest.mark.parametrize("save_path", ["./test_temp", None])
@pytest.mark.parametrize("fmt", ["coco"])
@pytest.mark.parametrize("has_split", [True])
def test_single_image(root, valid_name_idx, save_path, fmt):
    annotation = Annotation(root, format=fmt)
    vis = annotation.visualizer(split="valid")
    index = valid_name_idx[2]
    name = valid_name_idx[1]
    valid_name = valid_name_idx[0]

    if name and valid_name:
        name = "c4dfe80298624779.jpg"
    elif name:
        name = "c4dfe80298624779_rand.jpg"

    if not valid_name:
        with pytest.warns(UserWarning, match=f"{name} not found in the dataset. Please check"):
            _ = vis.show_image(index=index, name=name, render="pil")
    else:
        _ = vis.show_image(index=index, name=name, save_path=save_path, render="pil")
        if save_path is None:
            assert not Path("./test_temp").exists()
        else:
            img_path = f"{save_path}/{name}" if name is not None else save_path
            assert Path(img_path).exists()
            _ = vis.show_batch(num_imgs=1, save_path=save_path)
        rmtree(save_path, ignore_errors=True)
