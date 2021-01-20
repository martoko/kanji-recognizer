import argparse
import math
import os
import pathlib
import time
import random
from typing import Any, List, Optional

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import kanji
from datasets import KanjiRecognizerGeneratedDataset, RecognizerTestDataset


class LitRecognizer(pl.LightningModule):
    def __init__(self, character_set_name, learning_rate, **kwargs):
        super().__init__()

        self.character_set = kanji.character_sets[character_set_name]

        # Set up model
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv18 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.linear = nn.Linear(256 * 2 * 2, len(self.character_set))

        # Copy input to hparms
        self.save_hyperparameters()

        # Set up accuracy loggers
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = F.relu(self.conv1(x))

        before = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x += before

        before = x
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x += before

        before = x
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x += before

        before = x
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x += before

        before = x
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x += before

        x = x.view(-1, 256 * 2 * 2)
        x = self.linear(x)

        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

    def loss(self, images, labels):
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)

        self.log('train/loss', loss)
        self.log('train/acc', self.accuracy(logits, labels))

        return loss

    def validation_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)
        predictions = torch.argmax(logits, 1)

        self.log('valid/loss', loss)
        self.log('valid/acc', self.accuracy(logits, labels))

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        dummy_input = torch.zeros(1, 3, 32, 32, device=self.device)
        filename = f'model_{str(self.logger.step).zfill(5)}.onnx'
        torch.onnx.export(self, dummy_input, filename)
        wandb.save(filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log({
            'valid/logits': wandb.Histogram(flattened_logits.to('cpu'))
        })

    def test_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)

        self.log('test/loss', loss)
        self.log('test/acc', self.accuracy(logits, labels))

    def test_epoch_end(self, test_step_outputs):
        dummy_input = torch.zeros(1, 3, 32, 32, device=self.device)
        filename = 'model_final.onnx'
        torch.onnx.export(self, dummy_input, filename)
        wandb.save(filename)


class ImagePredictionLogger(pl.Callback):
    def __init__(self, samples, sample_count=32):
        super().__init__()
        images, labels = samples
        self.images = images[:sample_count]
        self.labels = labels[:sample_count]

    def on_validation_epoch_end(self, trainer, pl_module):
        images = self.images.to(device=pl_module.device)

        logits = pl_module(images)
        predictions = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            'examples': [wandb.Image(
                image,
                caption=f'Prediction: {prediction}, Label: {label}'
            ) for image, prediction, label in zip(images, predictions, self.labels)]
        })


class GaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class KanjiRecognizerDataModule(pl.LightningDataModule):
    def __init__(self, test_folder, fonts_folder, character_set_name, noise, batch_size=128, **kwargs):
        super().__init__()
        self.test_folder = test_folder
        self.fonts_folder = fonts_folder
        self.character_set = kanji.character_sets[character_set_name]
        self.batch_size = batch_size

        adversarial_transform = transforms.Compose([
            transforms.RandomCrop((32, 32)),
            transforms.ColorJitter(*args.color_jitter),
            transforms.Lambda(lambda img: PIL.ImageOps.invert(img) if random.random() > 0.5 else img),
            transforms.ToTensor(),
            GaussianNoise(*noise)
        ])

        plain_transform = transforms.Compose([
            transforms.CenterCrop((28, 28)),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        self.train_transform = transforms.Compose([
            transforms.Lambda(
                lambda img: adversarial_transform(img) if random.random() > 0.1 else plain_transform(img)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            # recognizer: mean 0.8515397726034853, std: 0.2013882252859569
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(size=(32, 32), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size=(32, 32), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = KanjiRecognizerGeneratedDataset(
                self.fonts_folder, characters=self.character_set, transform=self.train_transform
            )
            self.val = RecognizerTestDataset(
                self.test_folder, characters=self.character_set, transform=self.val_transform
            )
        if stage == 'test' or stage is None:
            self.test = RecognizerTestDataset(
                self.test_folder, characters=self.character_set, transform=self.test_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to recognize kanji.")
    pl.Trainer.add_argparse_args(parser)
    parser.add_argument("-o", "--output-path", type=str, default="data/models/recognizer.pt",
                        help="save model to this path")
    parser.add_argument("-i", "--input-path", type=str, default=None,
                        help="load model from this path")
    parser.add_argument("-f", "--fonts-folder", type=str, default="data/fonts",
                        help="path to a folder containing fonts (default: data/fonts)")
    parser.add_argument("-e", "--test-folder", type=str, default="data/test",
                        help="path to a folder containing test images named after their label character (default: data/test)")
    parser.add_argument("-b", "--background-images-folder", type=str, default="data/background-images",
                        help="path to a folder containing background images (default: data/background-images)")
    parser.add_argument("-B", "--batch-size", type=int, default=128,
                        help="the size of the batch used on each training step (default: 128)")
    parser.add_argument("-t", "--training-time", type=float, default=10,
                        help="amount of minutes to train the network (default: 10)")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the the optimizer (default: 1e-4)")
    parser.add_argument("-j", "--color-jitter", nargs='+', type=float, default=[0.1, 0.1, 0.1, 0.1],
                        help="brightness, contrast, saturation, hue passed onto the color jitter transform (default: 0.1, 0.1, 0.1, 0.1)")
    parser.add_argument("-n", "--noise", nargs='+', type=float, default=[0, 0.0005],
                        help="mean, std of gaussian noise transform (default: 0, 0.0005)")
    parser.add_argument("-c", "--character-set-name", type=str, default="frequent_kanji_plus",
                        help="name of characters to use (default: frequent_kanji_plus)")
    parser.add_argument("-F", "--log-frequency", type=float, default=60,
                        help="how many seconds between logging (default: 60)")
    args = parser.parse_args()

    datamodule = KanjiRecognizerDataModule(**vars(args))
    datamodule.setup()
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=WandbLogger(project='lit-qanji')
    )
    model = LitRecognizer(**vars(args))
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path=None)
