import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from recognizer.data.data_module import RecognizerDataModule
from recognizer.model import KanjiRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to recognize kanji.")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("-r", "--resume", type=str, default=False,
                        help="resumes a previous run given a run id or run path")
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="the size of the batch used on each training step (default: 128)")
    parser.add_argument("--data-folder", type=str, default="data",
                        help="path to a folder containing sub-folders with train/val/test data (default: data)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the the optimizer (default: 1e-4)")
    parser.add_argument("--character-set-name", type=str, default="frequent_kanji_plus",
                        help="name of characters to use (default: frequent_kanji_plus)")
    parser.add_argument("-w", "--num-workers", type=int, default=0,
                        help="number of workers to apply to data loading (default: 0)")

    parser.add_argument("-j", "--color-jitter", nargs='+', type=float, default=[0.1, 0.1, 0.1, 0.1],
                        help="brightness, contrast, saturation, hue passed onto the color jitter transform (default: 0.1, 0.1, 0.1, 0.1)")
    parser.add_argument("-n", "--noise", nargs='+', type=float, default=[0, 0.0005],
                        help="mean, std of gaussian noise transform (default: 0, 0.0005)")
    args = parser.parse_args()

    datamodule = RecognizerDataModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, logger=WandbLogger(entity="mb-haag-itu", log_model=True))
    model = KanjiRecognizer(**vars(args))
    trainer.fit(model, datamodule=datamodule)
