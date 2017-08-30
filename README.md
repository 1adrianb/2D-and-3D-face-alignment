# How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)

This repository implements a demo of the networks described in "How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)" paper. Please visit [our](https://www.adrianbulat.com) webpage or read bellow for instructions on how to run the code and access the dataset.

**Training code: <https://www.github.com/1adrianb/face-alignment-training>**

Note: If you are interested in a binarized version, capable of running on devices with limited resources please also check <https://github.com/1adrianb/binary-face-alignment> for a demo.

<p align='center'>
<img src='https://www.adrianbulat.com/images/image-z-examples.png' title='3D-FAN-Full example' style='max-width:600px'></img>
</p>

## Requirments

- Install the latest [Torch7](http://torch.ch/docs/getting-started.html) version (for Windows, please follow the instructions available [here](https://github.com/torch/distro/blob/master/win-files/README.md))
- Install python 2.7.x

### Lua packages

- [cutorch](https://github.com/torch/cutorch)
- [nn](https://github.com/torch/nn)
- [cunn](https://github.com/torch/cunn)
- [nngraph](https://github.com/torch/nngraph)
- [cudnn](https://github.com/soumith/cudnn.torch)
- [xlua](https://github.com/torch/xlua)
- [image](https://github.com/torch/image)
- [paths](https://github.com/torch/paths)
- [fb.python](https://github.com/facebook/fblualib/blob/master/fblualib/python/README.md)

### Python packages
- [numpy](http://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [dlib](https://pypi.python.org/pypi/dlib) (required for face detection, if no bbox are provided)

Please note that dlib performs poorly for faces found in challenging poses or difficult lighting conditions and it's provided only as a simple demo. For optimal performance we recommend using other deeplearning based face detection methods.

## Setup

Clone the github repository and install all the dependencies mentiones above.
```bash

git  clone https://github.com/1adrianb/2D-and-3D-face-alignment
cd 2D-and-3D-face-alignment
```

## Usage

In order to run the demo please download the required models available bellow and the associated data.

```bash
th main.lua
```

In order to see all the available options please run:

```bash
th main.lua --help
```

## Pretrained models

[2D-FAN](https://www.adrianbulat.com/downloads/FaceAlignment/2D-FAN-300W.t7) - trained on 300W-LP and finetuned on iBUG training set.

[3D-FAN](https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN.t7) - trained on 300W-LP

[2D-to-3D-FAN](https://www.adrianbulat.com/downloads/FaceAlignment/2D-to-3D-FAN.tar.gz) - trained on 300W-LP

[3D-FAN-depth](https://www.adrianbulat.com/downloads/FaceAlignment/3D-FAN-depth.t7) - trained on 300W-LP

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```

## Dataset

You can download the annotations alongside the images used by visiting [our page](https://www.adrianbulat.com/face-alignment).
