# scrungy-cats
Classifier for scrungy, drill, and other notable cat varieties.

## Installation

In a conda environment of your choosing (tested with Python 3.8):

```
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

## Usage

```
python train_model.py
```

### Usage via Docker

To create a Docker container for training:

```
docker run --gpus=all -it --mount type=bind,src=$PWD,dst=/app --shm-size 16G pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel bash
```

## Testing

First, install test dependencies if you do not already have them:

```
pip install flake8 mypy
```

Then run the tests (currently just static analysis):

```
flake8 && mypy
```
