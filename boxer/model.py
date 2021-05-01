import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F

from craft import CRAFT
from recognizer.data import character_sets


class KanjiBoxer(pl.LightningModule):
    def __init__(self, character_set_name, **kwargs):
        super().__init__()

        self.character_set = character_sets.character_sets[character_set_name]

        # Set up model
        self.model = CRAFT()

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

    def training_step(self, batch, batch_index):
        images, character_index, region_score = batch
        generated_region_score, _ = self(images)
        loss = F.mse_loss(generated_region_score, region_score)

        self.log('train/loss', loss)

        return loss
