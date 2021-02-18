import pytorch_lightning as pl
import torch
import torchvision
import wandb
from torch import optim
from torch.nn import functional as F

from recognizer.data import character_sets


class KanjiRecognizer(pl.LightningModule):
    def __init__(self, character_set_name, **kwargs):
        super().__init__()

        self.character_set = character_sets.character_sets[character_set_name]

        # Set up model
        self.model = torchvision.models.resnet152(num_classes=len(self.character_set))

        # Copy input to hparms
        self.save_hyperparameters()

        # Set up accuracy loggers
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

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

        self.log('val/loss', loss)
        accuracy = self.accuracy(logits, labels)
        self.log('val/acc', accuracy)
        if accuracy > 0.50 and self.train_dataloader().dataset.stage + 0.1 < 2:
            self.train_dataloader().dataset.stage += 0.1
        self.log('train/stage', self.train_dataloader().dataset.stage)

        return loss

    def test_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)

        self.log('test/loss', loss)
        self.log('test/acc', self.accuracy(logits, labels))


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
