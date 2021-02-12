from typing import Union, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from . import character_sets
from . import training_dataset
from . import validation_dataset


class RecognizerDataModule(pl.LightningDataModule):
    train: training_dataset.RecognizerTrainingDataset
    val: ConcatDataset
    test: ConcatDataset

    def __init__(self, data_folder: str, batch_size: int, character_set_name: str, **kwargs):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.batch_size = batch_size
        self.character_set = character_sets.character_sets[character_set_name]

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train = training_dataset.RecognizerTrainingDataset(
                data_folder=self.data_folder,
                character_set=self.character_set,
                transform=self.transform
            )
            self.val = validation_dataset.dataset_from_folder(
                data_folder=self.data_folder,
                character_set=self.character_set,
                transform=self.transform
            )

        if stage == 'test' or stage is None:
            self.test = validation_dataset.dataset_from_folder(
                data_folder=self.data_folder,
                character_set=self.character_set,
                transform=self.transform
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
