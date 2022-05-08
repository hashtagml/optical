Getting Started
###############

Usage
******


Declare imports and read the dataset
=====================================

.. code-block:: python

    from optical import Annotation

    root_dir = "/path/to/dataset"
    format = "coco"
    annotation = Annotation(root=root_dir, foramt=format)

optical expects the data to be organised in either of the following layouts:

.. code-block:: 

    root
    ├── images
    │ ├── train
    │ ├── val
    │ └── test
    └── annotations
      ├── train.json
      ├── val.json
      └── test.json

Note that for annotation formats which require individual annotations for each images (e.g., ``PASCAL VOC`` or ``Yolo``), 
the ``annotations`` directory should also contain the same sub-directories as in ``images``. The splits that do not have an annotation will be ignored.

If your data does not have a split to begin with, that's acceptable too. In that case the directory layout should be like below:

.. code-block::

    root
    ├── images
    │ ├── 1.jpg
    │ ├── 2.jpg
    │ ├── ...
    │ │
    │ └── 100.jpg
    │
    └── annotations
      └── label.json

Tha name of the annotation file is not important in this case. But, if your format requires individual annotation files, the annotation files must have the identical name with that of the image.

EDA
====

**Check data distribution:**

.. code-block:: python

    annotation.describe()

    | split | images | annotations | categories |
    | ----- | ------ | ----------- | ---------- |
    | train | 729    | 1121        | 3          |
    | valid | 250    | 322         | 3          |

**Plot label distribution:**

.. code-block:: python

    annotation.show_distribution()

.. image:: ../../assets/show_dist.png

**Scatter Plot of bounding box width and height:**

.. code-block:: python

    annotation.bbox_scatter()

.. image:: ../../assets/bbox_scatter.png

Visualization
==============

**Batch Visualization:**

.. code-block:: python

    vis = annotation.visualizer(img_size=256)
    vis.show_batch()

.. image:: ../../assets/vis_batch.png

**Filters:**

We can apply below filters before visualizing the annotations. 

.. code-block:: python

    vis.show_batch(num_imgs=9,only_without_labels=True) # To only display images without annotations
    vis.show_batch(num_imgs=9,only_with_labels=True) # To only display images with annotations
    vis.show_batch(num_imgs=9,filter_categories=["WBC","RBC"]) # To only display images with given category annotations.

**Video Visualization:**

We can visualize whole dataset as a video which can be played in the jupyter notebook itself. This is intended for smaller datasets.

.. code-block:: python

    image_time = 1 #Meaning each image will be displayed for a second.
    vis.show_video(image_time = image_time)

Splitting Dataset
==================

.. code-block:: python

    splits = annotation.train_test_split(test_size = 0.2, stratified = True)
    splits.save("/path/to/output/dir")

Exporting to other formats
===========================

.. code-block:: python

    annotation.export(to = "yolo")













