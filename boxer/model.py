import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
import wandb

from boxer.craft import CRAFT
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

    def training_epoch_end(self, *args):
        images, character_index, region_scores = next(iter(self.train_dataloader()))
        generated_region_scores, _ = self(images)
        self.log('train/images', [wandb.Image(x) for x in images[:8]])
        self.log('train/region_scores', [wandb.Image(x) for x in
            region_scores[:8]])
        self.log('train/generated_region_scores', [wandb.Image(x) for x in
            generated_region_scores[:8]])
       # iwandb.log({"train/failure_cases": [wandb.Image(
       #          case["image"],
       #          caption=f"Prediction: {case['prediction']} Truth: {case['label']}"
       #      ) for case in sorted(failure_cases, key=lambda item: item['confidence'])[:1]]}, commit=False)

        dataset = self.train_dataloader().dataset
        dataset.stage += 0.1
        self.log('train/stage', dataset.stage, prog_bar=True)
