import pytorch_lightning as pl
import torch
import torchvision
import wandb
from torch import optim
from torch.nn import functional as F
from vit_pytorch import ViT

from recognizer.data import character_sets


class KanjiRecognizer(pl.LightningModule):
    def __init__(self, character_set_name, model_type="resnet"):
        super().__init__()

        self.character_set = character_sets.character_sets[character_set_name]

        # Set up model
        if model_type == "resnet":
            self.model = torchvision.models.resnet152(num_classes=len(self.character_set))
        if model_type == "ViT":
            self.model = ViT(
                image_size=128,
                # Number of patches. image_size must be divisible by patch_size.
                # The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16.
                patch_size=16,
                num_classes=len(self.character_set),
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )

        # Copy input to hparms
        self.save_hyperparameters()

        # Set up accuracy loggers
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

    def loss(self, images, labels):
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_index):
        images, labels, _ = batch
        logits, loss = self.loss(images, labels)

        self.log('train/loss', loss)
        self.log('train/acc_step', self.train_accuracy(F.softmax(logits, dim=1), labels))

        return loss

    def validation_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)

        self.log('val/loss', loss, prog_bar=True)
        accuracy = self.val_accuracy(F.softmax(logits), labels)
        self.log('val/acc', accuracy)

        return loss

    def training_epoch_end(self, *args):
        dataset = self.train_dataloader().dataset
        self.log('train/acc_epoch', self.train_accuracy, prog_bar=True)
        if self.train_accuracy.compute() > 0.95:
            dataset.stage += 0.1
        self.log('train/stage', dataset.stage, prog_bar=True)

    def test_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)

        self.log('test/loss', loss)
        self.log('test/acc', self.test_accuracy(F.softmax(logits), labels))


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
