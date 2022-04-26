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

## Testing

First, install test dependencies if you do not already have them:

```
pip install flake8 mypy
```

Then run the tests (currently just static analysis):

```
flake8 && mypy
```
