.. optical documentation master file, created by
   sphinx-quickstart on Thu Jan 14 19:22:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/banner.png
   :width: 800
   :align: center

Optical
=======

Optical aims to be the tool right next to your hand when you work on a computer vision project. Optical does not help you with training your deep learning models neither does it perform any fancy augmentation. It's rather about making life easier for you by automating all the routine tasks that you often go through before you can actually start training your model.


Features
__________

* **Object detection**

      Object detection is one of the mainstream computer vision tasks. However, when it comes to training an object detection model, there are a variety of data formats that one has to deal with for different models e.g. COCO, PASCAL VOC, Yolo and so on. Optical provides a simple interface to convert back and forth between these annotation formats and also perform a bunch of exploratory data analysis (EDA) along the way.

      At present we support conversion to and from the following formats 🚀

      * `COCO <https://cocodataset.org/#format-data>`_
      * `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data>`_
      * `Yolo <https://github.com/AlexeyAB/darknet>`_
      * `TFrecord <https://www.tensorflow.org/tutorials/load_data/tfrecord>`_
      * `SageMaker Manifest <https://docs.aws.amazon.com/sagemaker/latest/dg/augmented-manifest.html>`_
      * `CreateML <https://hackernoon.com/how-to-label-data-create-ml-for-object-detection-82043957b5cb>`_
      * CSV

      See this :doc:`turorial↩ <overview>` to get started.

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:
   
   Installation <install>
   Getting Started <overview>

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   Coding Requirements <coding_requirements>
   Issues <issues>
   Tests <tests>
   Setting up a Development Environment <setupdev>
   Documentation <builddocs>
   Pull Requests <pull_requests>

.. toctree::
   :maxdepth: 1
   :Caption: Advanced
   :hidden:

   API Documentation <api>
   Changelog <changelog>
