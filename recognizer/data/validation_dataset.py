import glob
import os
import pathlib
import random

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, ConcatDataset

from . import character_sets

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RecognizerValidationSingleImageDataset(Dataset):
    def __init__(self, path, characters, transform=None):
        super().__init__()
        self.transform = transform
        self.image_path = pathlib.Path(path)
        self.max_translation = int(self.image_path.with_suffix(".txt").read_text())
        self.character = self.image_path.parent.name
        self.characters = characters

    def __len__(self):
        if self.character not in self.characters:
            return 0
        return (self.max_translation * 2) ** 2

    @staticmethod
    def load(path: str) -> Image.Image:
        with open(path, 'rb') as file:
            return Image.open(file).convert('RGB')

    def __getitem__(self, index) -> T_co:
        sample = Image.new('RGB', (128, 128), (255, 255, 255))

        image = self.load(self.image_path)
        x = self.max_translation - index % (self.max_translation * 2)
        y = self.max_translation - index // (self.max_translation * 2)
        sample.paste(image, (x, y))

        label = self.characters.index(self.character)

        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label


def dataset_from_folder(data_folder, character_set, transform=None):
    paths = glob.glob(os.path.join(data_folder, "free-kanji", '**/*.png'), recursive=True)
    datasets = [RecognizerValidationSingleImageDataset(path, character_set, transform) for path in paths]
    dataset = ConcatDataset(datasets)
    assert len(dataset) > 0
    return dataset


if __name__ == '__main__':
    def generate(dataset, count=None):
        # Remove existing data
        files = glob.glob(f"generated/validation/*")
        for file in files:
            os.remove(file)
        pathlib.Path(f"generated/validation").mkdir(parents=True, exist_ok=True)

        # Save new data
        for i in random.sample(range(len(dataset)), count):
            dataset[i][0].save(f"generated/validation/{i}.png")


    dataset = dataset_from_folder("data", character_set=character_sets.frequent_kanji_plus)
    print(f"size: {len(dataset)}")
    generate(dataset, count=300)
