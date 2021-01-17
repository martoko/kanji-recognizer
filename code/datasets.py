import glob
import os
import random
import pathlib
import shutil
from random import randint
from typing import *
import PIL
import numpy as np
from PIL import ImageFont, Image, ImageFile, ImageDraw, ImageStat
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data.dataset import T_co
import kanji

ImageFile.LOAD_TRUNCATED_IMAGES = True


def font_paths(folder):
    return glob.glob(os.path.join(folder, '**/*.ttf'), recursive=True) + \
           glob.glob(os.path.join(folder, '**/*.ttc'), recursive=True)


def background_images(folder):
    return [
        Image.open(os.path.join(folder, name))
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    ]


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def random_white_color():
    return randint(230, 255), randint(230, 255), randint(230, 255)


def random_black_color():
    return randint(0, 50), randint(0, 50), randint(0, 50)


class BoxerDataset(IterableDataset):
    def __init__(self, fonts_folder: str, background_image_folder: str, transform=None):
        super(BoxerDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.jouyou_kanji

    def generate(self):
        character = self.characters[self.character_index]
        font_path = random.choice(self.font_files)
        background = random.choice(self.background_images)
        font_size = randint(12, 16)
        font = ImageFont.truetype(font_path, font_size)

        bg_left = randint(0, background.width - 10)
        bg_right = randint(bg_left + 5, background.width)
        bg_top = randint(0, background.height - 10)
        bg_bottom = randint(bg_top + 5, background.height)

        left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
        sample = background.resize(
            (32, 32), resample=PIL.Image.NEAREST, box=(bg_left, bg_top, bg_right, bg_bottom)
        )

        center_x = 32 / 2 - randint(0, right)
        center_y = 32 / 2 - randint(0, bottom)
        label = [center_x, center_y, center_x + right, center_y + bottom]

        drawing = ImageDraw.Draw(sample)
        drawing.text((center_x, center_y), character, font=font,
                     fill=random.choice([random_color(), random_black_color(), random_white_color()]),
                     anchor='lt', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate()

    def __getitem__(self, index) -> T_co:
        return self.generate()


class RecognizerGeneratedDataset(IterableDataset):
    # TODO: Rotate/morph/skew
    # TODO: Other characters as part of background?
    # TODO: Add a small random padding/cropping to be more resistant to faulty cropping
    # TODO: Add random lines to image, also in same color as text
    # TODO: Legible text is more common than completely illegible text
    # TODO: Random noise/color noise
    def __init__(self, fonts_folder: str, background_image_folder: str, transform=None):
        super(RecognizerGeneratedDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.jouyou_kanji
        self.id = 'recognizer-1'

    def generate(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_path = random.choice(self.font_files)
        background = random.choice(self.background_images)
        background = Image.new('RGB', (32, 32), color=random_white_color())
        font_size = randint(12, 16)
        font = ImageFont.truetype(font_path, font_size)

        bg_left = randint(0, background.width - 10)
        bg_right = randint(bg_left + 5, background.width)
        bg_top = randint(0, background.height - 10)
        bg_bottom = randint(bg_top + 5, background.height)

        # TODO: Jitter

        left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
        if right == 0 or bottom == 0:
            print(f"{character} is missing from {font}")
        sample = background.resize(
            (right, bottom), resample=PIL.Image.NEAREST, box=(bg_left, bg_top, bg_right, bg_bottom)
        )
        drawing = ImageDraw.Draw(sample)
        drawing.text((0, 0), character, font=font, fill=random_color(), anchor='lt', language='ja')
        sample = sample.resize((32, 32), resample=PIL.Image.NEAREST)

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate()

    def __getitem__(self, index) -> T_co:
        return self.generate()


class RecognizerTestDataset(Dataset):
    def __init__(self, path, transform=None):
        super(RecognizerTestDataset).__init__()
        self.id = 'recognizer-test-1'
        self.characters = kanji.jouyou_kanji
        self.transform = transform
        paths = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
        characters = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        self.samples = [(path, character) for path, character
                        in zip(paths, characters)
                        if character in self.characters]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load(path: str) -> Image.Image:
        with open(path, 'rb') as file:
            return Image.open(file).convert('RGB')

    def __getitem__(self, index) -> T_co:
        path, character = self.samples[index]
        sample = self.load(path)
        label = self.characters.index(character)

        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label


class HiraganaDataset(IterableDataset):
    def __init__(self, fonts_folder: str, background_image_folder: str, transform=None):
        super(HiraganaDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.simple_hiragana

    def generate(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_path = random.choice(self.font_files)
        sample = Image.new('RGB', (32, 32), color=random_white_color())
        font_size = randint(10, 31)
        font = ImageFont.truetype(font_path, font_size)

        left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
        x = randint(0, 32 - right)
        y = randint(0, 32 - bottom)
        drawing = ImageDraw.Draw(sample)
        drawing.text((x, y), character, font=font, fill=random_black_color(), anchor='lt', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate()

    def __getitem__(self, index) -> T_co:
        return self.generate()


# TODO: Shuffle character order
if __name__ == '__main__':
    def generate(dataset, count=50):
        if pathlib.Path(f"data/generated/{dataset.id}").exists():
            shutil.rmtree(f"data/generated/{dataset.id}")
        pathlib.Path(f"data/generated/{dataset.id}").mkdir(parents=True, exist_ok=True)
        iterator = iter(dataset)
        for i in range(count):
            sample, label = next(iterator)
            sample.save(f"data/generated/{dataset.id}/{i}.png")


    def normalization_data(dataset, name, count=1000):
        pixels = np.array([])
        iterator = iter(dataset)
        for i in range(count):
            sample, label = next(iterator)
            pixels = np.append(pixels, np.array(sample) / 255)
            if i % (round(count / 100)) == 0:
                print(f"{round(i / count * 100)}% {i}/{count}\r", end='')
        print(f"{name}: mean {pixels.mean()}, std: {pixels.std()}")


    normalization_data(RecognizerGeneratedDataset("data/fonts", "data/background-images"), "recognizer")
    # normalization_data(BoxerDataset("data/fonts", "data/background-images"), "boxer")
    # normalization_data(HiraganaDataset("data/fonts", "data/background-images"), "hiragana")

    generate(RecognizerGeneratedDataset("data/fonts", "data/background-images"))
    # generate(BoxerDataset("data/fonts", "data/background-images"), "boxer")
    # generate(HiraganaDataset("data/fonts", "data/background-images"), "hiragana")
