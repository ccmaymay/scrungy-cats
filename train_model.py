#!/usr/bin/env python3

"""
This code is based on the "Finetuning Torchvision Models" tutorial at:
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

import copy
import csv
import logging
from collections.abc import Sized
from functools import partial
from os import PathLike
from pathlib import Path
from typing import (
    cast, Any, Callable, Dict, Iterable, List, Literal, Optional, NamedTuple, Tuple, Union,
)

from PIL import Image  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import models, transforms  # type: ignore

Device = Union[torch.device, str]
Phase = Literal['train', 'val']
Transform = Callable[[Any], Any]
ImageTransform = Callable[[Image.Image], Any]
LabelsTransform = Callable[[List[int]], Any]

PHASES: Tuple[Phase, Phase] = ('train', 'val')
LABELS_FILENAME = 'labels.csv'
DEFAULT_DATA_DIR = 'data'
DEFAULT_INPUT_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 25
DEFAULT_NUM_DATA_WORKERS = 4


class SampleInfo(NamedTuple):
    labels: List[int]
    path: PathLike


def get_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def do_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: Any,
             optimizer: optim.Optimizer, is_train: bool,
             device: Optional[Device] = None) -> Tuple[float, float]:
    if device is None:
        device = get_default_device()

    total_loss = 0.
    total_error = 0.

    if is_train:
        model.train()
    else:
        model.eval()

    # Iterate over data
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(is_train):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward, optimize only if in training phase
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_error += mean_squared_error(labels.data.numpy(), outputs.detach().cpu().numpy())

    epoch_loss = total_loss / len(cast(Sized, dataloader.dataset))
    epoch_error = total_error / len(cast(Sized, dataloader.dataset))

    return (epoch_loss, epoch_error)


def train_model(model: nn.Module, dataloaders: Dict[Phase, torch.utils.data.DataLoader],
                criterion: Any, optimizer: optim.Optimizer, num_epochs: int = DEFAULT_NUM_EPOCHS,
                device: Optional[Device] = None) -> nn.Module:
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    best_params = copy.deepcopy(model.state_dict())
    best_error = 0.

    for epoch in range(num_epochs):
        logging.info(f'Starting epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in PHASES:
            (epoch_loss, epoch_error) = do_epoch(
                model, dataloaders[phase], criterion, optimizer, phase == 'train', device)

            logging.info(f'{phase} loss: {epoch_loss:.3f}, error: {epoch_error:.3f}')

            # deep copy the model
            if phase == 'val':
                if epoch_error < best_error:
                    best_error = epoch_error
                    best_params = copy.deepcopy(model.state_dict())

    logging.info('Best val error: {:.3f}'.format(best_error))

    # load best model weights
    model.load_state_dict(best_params)
    return model


def set_parameter_requires_grad(model: nn.Module, feature_extract: bool):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def make_densenet_model(num_classes: int, feature_extract: bool,
                        use_pretrained: bool = True) -> nn.Module:
    model = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, num_classes), nn.Sigmoid())

    return model


def make_sgd_optimizer(params_to_update: Iterable[nn.Parameter],
                       lr=0.001, momentum=0.9) -> optim.Optimizer:
    return optim.SGD(params_to_update, lr=lr, momentum=momentum)


def get_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> Dict[Phase, Transform]:
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def multi_hot(labels: List[int], num_classes: int) -> torch.Tensor:
    return torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(labels), value=1.)


def get_target_transform(num_classes: int) -> Transform:
    return transforms.Lambda(partial(multi_hot, num_classes=num_classes))


class CSVLabeledImageDataset(torch.utils.data.Dataset):
    sample_infos: List[SampleInfo]
    classes: List[str]
    transform: Optional[ImageTransform]
    target_transform: Optional[LabelsTransform]

    def __init__(self,
                 data_dir: PathLike,
                 transform: Optional[ImageTransform] = None,
                 target_transform: Optional[LabelsTransform] = None):
        labels_path = Path(data_dir) / LABELS_FILENAME
        logging.info(f'Loading labels from {labels_path}')
        with open(labels_path) as f:
            self.classes = []
            self.sample_infos = []
            for row in csv.reader(f):
                if self.classes:
                    row_filename = row[0]
                    row_labels = [
                        self.classes.index(class_name)
                        for (class_name, v) in zip(self.classes, row[1:])
                        if int(v)
                    ]
                    row_path = Path(data_dir) / row_filename
                    self.sample_infos.append(SampleInfo(row_labels, row_path))
                else:
                    self.classes = row[1:]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample_info = self.sample_infos[index]

        sample = self.load_image(sample_info.path)
        if self.transform is not None:
            sample = self.transform(sample)

        labels = sample_info.labels
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return (sample, labels)

    def __len__(self) -> int:
        return len(self.sample_infos)

    def load_image(self, path: PathLike) -> Image.Image:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')


def train_model_on_csv_labeled_images(
        make_model: Optional[Callable[[int, bool], nn.Module]] = None,
        make_optimizer: Optional[Callable[[Iterable[nn.Parameter]], optim.Optimizer]] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        feature_extract: bool = True,
        data_dir: PathLike = cast(PathLike, DEFAULT_DATA_DIR),
        num_data_workers: int = DEFAULT_NUM_DATA_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        device: Optional[Device] = None) -> nn.Module:

    if make_model is None:
        make_model = make_densenet_model

    if make_optimizer is None:
        make_optimizer = make_sgd_optimizer

    if device is None:
        device = get_default_device()

    logging.info('Initializing data...')
    datasets = dict(
        (phase, CSVLabeledImageDataset(Path(data_dir) / phase))
        for phase in PHASES)
    transforms = get_transforms(input_size)
    classes = datasets['train'].classes
    num_classes = len(classes)
    for phase in PHASES:
        datasets[phase].transform = transforms[phase]
        datasets[phase].target_transform = get_target_transform(num_classes)
    classes_str = ' '.join(f'[{i}] {c}' for (i, c) in enumerate(classes))
    logging.info(f'Found {num_classes} classes: {classes_str}')

    dataloaders = dict(
        (phase, torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=True,
                                            num_workers=num_data_workers))
        for phase in PHASES)

    logging.info('Loading model...')
    model = make_model(num_classes, feature_extract).to(device)

    params_to_update: Iterable[nn.Parameter]
    if feature_extract:
        logging.info('Extracting features from existing model: updating last layer only')
        params_to_update = [param for param in model.parameters() if param.requires_grad]
    else:
        logging.info('Fine-tuning existing model: updating all layers')
        params_to_update = model.parameters()

    optimizer = make_optimizer(params_to_update)
    criterion = nn.BCELoss()

    logging.info('Beginning training...')
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs,
                        device=device)
    logging.info('Training complete.')

    return model


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
                        level=logging.INFO)
    train_model_on_csv_labeled_images()
