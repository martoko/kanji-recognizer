import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from recognizer.data.data_module import RecognizerDataModule
from recognizer.model import KanjiRecognizer

if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'data_folder': 'data',
        'learning_rate': 1e-3,
        'character_set_name': 'top_100_kanji',
        'num_workers': 0,
        'model_type': 'resnet',
        'logger': True,
        # 'logger': WandbLogger(entity="mb-haag-itu", log_model=True)
    }

    datamodule = RecognizerDataModule(**config)
    trainer = pl.Trainer(
        gpus=None,
        limit_train_batches=100,
        val_check_interval=100,
        max_epochs=10,
        stochastic_weight_avg=True,
        logger=config['logger'],
        auto_lr_find=True
    )
    tuner = trainer.tuner
    model = KanjiRecognizer(**config)

    lr_finder = tuner.lr_find(model, datamodule=datamodule, num_training=20)
    fig = lr_finder.plot()
    fig.show()
    suggested_lr = lr_finder.suggestion()
    print(suggested_lr)
    # trainer.tune(model, datamodule=datamodule)
    # print(model.learning_rate)
    # trainer.fit(model, datamodule=datamodule)
