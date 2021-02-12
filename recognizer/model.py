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
        predictions = torch.argmax(logits, 1)

        self.log('val/loss', loss)
        self.log('val/acc', self.accuracy(predictions, labels))

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        # dummy_input = torch.zeros(1, 3, 32, 32, device=self.device)
        # filename = f'model_{str(self.logger.step).zfill(5)}.onnx'
        # torch.onnx.export(self, dummy_input, filename)
        # wandb.save(filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log({
            'val/logits': wandb.Histogram(flattened_logits.to('cpu'))
        })

    def test_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)
        predictions = torch.argmax(logits, 1)

        self.log('test/loss', loss)
        self.log('test/acc', self.accuracy(predictions, labels))

    def test_epoch_end(self, test_step_outputs):
        # dummy_input = torch.zeros(1\, 3, 32, 32, device=self.device)
        # filename = 'model_final.onnx'
        # torch.onnx.export(self, dummy_input, filename)
        # wandb.save(filename)
        pass


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
