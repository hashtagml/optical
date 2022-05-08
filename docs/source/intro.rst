Introduction
============

What is optical?
----------------
Object detection is one of the mainstream computer vision tasks. However, when it comes to training an object detection model, there is a variety of formats that one has to deal with for different models e.g. COCO, PASCAL VOC, Yolo and so on. optical provides a simple interface to convert back and forth between these annotation formats and also perform a bunch of exploratory data analysis (EDA) on these datasets regardless of their source format.

🌟 At present we support the following formats:

 * `COCO <https://cocodataset.org/#format-data>`_
 * `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data>`_
 * `Yolo <https://github.com/AlexeyAB/darknet>`_
 * `TFrecord <https://www.tensorflow.org/tutorials/load_data/tfrecord>`_
 * `SageMaker Manifest <https://docs.aws.amazon.com/sagemaker/latest/dg/augmented-manifest.html>`_
 * `CreateML <https://hackernoon.com/how-to-label-data-create-ml-for-object-detection-82043957b5cb>`_
 * CSV
