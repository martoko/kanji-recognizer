import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from boxer.model import KanjiBoxer
from recognizer.data.data_module import RecognizerDataModule

if __name__ == "__main__":
    args = {
      "batch_size": 4,
      "data_folder":"data",
      "learning_rate": 6.9e-05,
      "character_set_name": "frequent_kanji_plus",
      "num_workers": 0,
      "model_type": "resnet",
      "accumulate_grad_batches": 1
    }

    datamodule = RecognizerDataModule(**args)
    datamodule.setup()
    datamodule.train.stage = 3
    trainer = pl.Trainer(
      limit_train_batches=3 * args["accumulate_grad_batches"],
      val_check_interval=3 * args["accumulate_grad_batches"],
      max_epochs=10,
      stochastic_weight_avg=True,
      accumulate_grad_batches=args["accumulate_grad_batches"],
      gpus=0,
      logger=WandbLogger(entity="mb-haag-itu", log_model=True, project="kanji-boxer")
    )
    model = KanjiBoxer(**args)
    trainer.fit(model, datamodule=datamodule)
