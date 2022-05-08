<div align="center">

# Optical
![ci build](https://github.com/hashtagml/optical/actions/workflows/build.yaml/badge.svg)
[![codecov](https://codecov.io/gh/hashtagml/optical/branch/main/graph/badge.svg?token=EC26I5HFQH)](https://codecov.io/gh/hashtagml/optical)
![Python Version](https://img.shields.io/pypi/pyversions/optical)
[![Documentation Status](https://readthedocs.org/projects/optical/badge/?version=latest)](https://optical.readthedocs.io/en/latest/?badge=latest)
![GitHub all releases](https://img.shields.io/github/downloads/hashtagml/optical/total)
![PyPI - License](https://img.shields.io/pypi/l/optical)
[![PyPI version](https://badge.fury.io/py/optical.svg)](https://badge.fury.io/py/optical)
<!-- [![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat)](#contributors-) -->

<p align="center"><img align="centre" src="assets/banner-dark.png" alt="logo" width = "650"></p>

Optical aims to be the tool right next to your hand when you work on a computer vision project. Optical does not help you with training your deep learning models neither does it perform any fancy augmentation. It's rather about making life easier for you by automating all the routine tasks that you often go through before you can actually start training your model.


<br/>
<br/>

</div>

## Where is optical useful?

Object detection is one of the mainstream computer vision tasks. However, when it comes to training an object detection model, there is a variety of formats that one has to deal with for different models e.g. `COCO`, `PASCAL VOC`, `Yolo` and so on. `optical` provides a simple interface to convert back and forth between these annotation formats and also perform a bunch of exploratory data analysis (EDA) on these datasets regardless of their source format.

At present we support the following formats 🚀:
- [COCO](https://cocodataset.org/#format-data)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)
- [Yolo](https://github.com/AlexeyAB/darknet)
- [TFrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [SageMaker Manifest](https://docs.aws.amazon.com/sagemaker/latest/dg/augmented-manifest.html)
- [CreateML](https://hackernoon.com/how-to-label-data-create-ml-for-object-detection-82043957b5cb)
- CSV


## Installation

`optical` is available in PyPi and can be installed with `pip` like so.

```sh
pip install --upgrade optical
```

For conversion to (or from) `TFrecord`, please install the `tensorflow` extra:
```sh
pip install optical[tensorflow]
```

for visualization of images in [mediapy](https://github.com/google/mediapy) format, you need to have [ffmpeg](https://ffmpeg.org/download.html) installed in your system.


## Getting Started

See this [quick started guide](https://optical.readthedocs.io/en/latest/overview.html) to get off the ground with optical.
## Contributing

### Work in local environment:

1. Fork the repo
2. install poetry:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. work on virtual environment:
   ```sh
   conda create -n optical python=3.8 pip
   ```

4. install the dependencies and the project in editable mode
   ```sh
   poetry install
   ```
5. Make your changes as required. Please use appropriate use of docstrings (we follow [Google style docstring](https://google.github.io/styleguide/pyguide.html)) and try to keep your code clean.

6. Raise a pull request.

### Work inside the dev container:
If you are a Visual Studio Code user, you may choose to develop inside a container. The benefit is the container comes with all necessary settings and dependencies configured. You will need [Docker](https://www.docker.com/) installed in your system. You also need to have the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension enabled.

1. Open the project in Visual Studio Code. in the status bar, select open in remote container.

It will perhaps take a few minutes the first time you build the container.
