#!/usr/bin/env python3

"""
This code is based on the "Finetuning Torchvision Models" tutorial at:
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

import copy
import csv
import logging
import os
from collections.abc import Sized
from functools import partial
from os import PathLike
from pathlib import Path
from typing import (
    cast, Any, Callable, Dict, Iterable, List, Literal, Optional, NamedTuple, Tuple, Union,
)

import numpy as np
from PIL import Image  # type: ignore
from sklearn.metrics import f1_score, mean_squared_error, classification_report  # type: ignore
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
FILENAME_FIELD = 'filename'
WEIGHTS_FILENAME = 'best.pt'
CLASSES_FILENAME = 'classes.txt'
DEFAULT_DATA_DIR = 'data'
DEFAULT_SAVE_DIR = 'model'
DEFAULT_INPUT_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_DATA_WORKERS = 4
SCORE_AVERAGING = 'macro'
CONFIDENCE_THRESHOLDS = tuple(i/100 for i in range(1, 100))
LEARNING_RATE = 0.001
MOMENTUM = 0.9


class SampleInfo(NamedTuple):
    labels: List[int]
    path: PathLike


class EpochResults(NamedTuple):
    loss: float
    mse: float
    confidence_threshold: float
    f1: float
    report_str: str

    def __str__(self) -> str:
        return (
            f'Loss: {self.loss:.3f}\n'
            f'MSE: {self.mse:.3f}\n'
            f'Metrics for threshold classifier f(x) > {self.confidence_threshold:.2f}:\n'
            f'{self.report_str}'
        )


def get_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def do_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: Any,
             optimizer: optim.Optimizer, is_train: bool,
             confidence_threshold: Optional[float] = None,
             device: Optional[Device] = None) -> EpochResults:
    if device is None:
        device = get_default_device()

    total_loss = 0.
    total_labels: np.ndarray = np.zeros((0, 0))
    total_outputs: np.ndarray = np.zeros((0, 0))

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
        np_labels = labels.data.cpu().numpy()
        np_outputs = outputs.detach().cpu().numpy()
        if total_labels.shape[0] == 0 or total_outputs.shape[0] == 0:
            total_labels = np_labels
            total_outputs = np_outputs
        else:
            total_labels = np.concatenate((total_labels, np_labels))
            total_outputs = np.concatenate((total_outputs, np_outputs))

    if is_train:
        f1 = [
            f1_score(
                total_labels, total_outputs > t, average=SCORE_AVERAGING, zero_division=0,
            )
            for t in CONFIDENCE_THRESHOLDS
        ]
        confidence_threshold_idx = max(enumerate(f1), key=lambda p: p[1])[0]
        confidence_threshold = CONFIDENCE_THRESHOLDS[confidence_threshold_idx]

    if confidence_threshold is None:
        raise Exception('Confidence threshold is required to compute metrics when not training')

    return EpochResults(
        loss=total_loss / len(cast(Sized, dataloader.dataset)),
        mse=mean_squared_error(total_labels, total_outputs),
        confidence_threshold=confidence_threshold,
        f1=f1_score(
            total_labels, total_outputs > confidence_threshold,
            average=SCORE_AVERAGING, zero_division=0,
        ),
        report_str=classification_report(
            total_labels, total_outputs > confidence_threshold,
            target_names=cast(CSVLabeledImageDataset, dataloader.dataset).classes,
            zero_division=0,
        ),
    )


def train_model(model: nn.Module, dataloaders: Dict[Phase, torch.utils.data.DataLoader],
                criterion: Any, optimizer: optim.Optimizer, num_epochs: int = DEFAULT_NUM_EPOCHS,
                device: Optional[Device] = None) -> nn.Module:
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    best_params = copy.deepcopy(model.state_dict())
    best_results: Optional[EpochResults] = None

    for epoch in range(num_epochs):
        logging.info(f'Starting epoch {epoch}/{num_epochs - 1}...')

        # Each epoch has a training and validation phase
        confidence_threshold: Optional[float] = None
        for phase in PHASES:
            epoch_results = do_epoch(
                model, dataloaders[phase], criterion, optimizer, phase == 'train',
                confidence_threshold=confidence_threshold,
                device=device)

            logging.info(f'{phase} results:\n{epoch_results}')

            if phase == 'train':
                # copy the best confidence threshold for validation
                confidence_threshold = epoch_results.confidence_threshold

            elif phase == 'val':
                # deep copy the model & results if the validation score has improved
                if best_results is None or epoch_results.f1 > best_results.f1:
                    best_params = copy.deepcopy(model.state_dict())
                    best_results = epoch_results

    if best_results is None:
        raise Exception(f'Produced no results after {num_epochs} epochs')

    logging.info(f'Best val results:\n{best_results}')

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
                       lr: float = LEARNING_RATE, momentum: float = MOMENTUM) -> optim.Optimizer:
    return optim.SGD(params_to_update, lr=lr, momentum=momentum)


def get_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> Dict[Phase, Transform]:
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.5, 1.)),
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
                 classes: Optional[List[str]] = None,
                 transform: Optional[ImageTransform] = None,
                 target_transform: Optional[LabelsTransform] = None):
        self.sample_infos = []

        labels_path = Path(data_dir) / LABELS_FILENAME
        logging.info(f'Loading labels from {labels_path}')
        with open(labels_path) as f:
            for row in csv.DictReader(f):
                if classes is None:
                    classes = sorted(k for k in row.keys() if k != FILENAME_FIELD)

                row_filename = row['filename']
                row_labels = [
                    i
                    for (i, class_name) in enumerate(classes)
                    if int(row[class_name])
                ]
                row_path = Path(data_dir) / row_filename
                self.sample_infos.append(SampleInfo(row_labels, row_path))

        if classes is not None:
            self.classes = classes
        else:
            raise Exception('Classes were not passed as parameters and could not be inferred.')

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


def save_classes(classes: List[str], path: PathLike):
    with open(path, mode='w') as f:
        for class_name in classes:
            f.write(f'{class_name}\n')


def train_model_on_csv_labeled_images(
        classes: Optional[List[str]] = None,
        save_dir: Optional[PathLike] = cast(PathLike, DEFAULT_SAVE_DIR),
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
    val_dataset = CSVLabeledImageDataset(Path(data_dir) / 'val', classes=classes)
    classes = val_dataset.classes
    num_classes = len(classes)
    datasets = {
        'train': CSVLabeledImageDataset(Path(data_dir) / 'train', classes=classes),
        'val': val_dataset,
    }
    transforms = get_transforms(input_size)
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

    if save_dir is not None:
        logging.info(f'Saving model to {save_dir}')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), Path(save_dir) / WEIGHTS_FILENAME)
        save_classes(classes, Path(save_dir) / CLASSES_FILENAME)

    return model


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Train classifier of images of cats as scrunge, drill, mlem, etc. '
                    '(as a multi-label classification task)',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--classes', type=str,
                        help='List of classes to predict as a comma-separated list')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
                        level=logging.INFO)
    train_model_on_csv_labeled_images(classes=args.classes.split(',') if args.classes else None)


if __name__ == '__main__':
    main()
