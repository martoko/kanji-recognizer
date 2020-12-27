import glob
import os
import random
from random import randint
from typing import *
import numpy as np

import PIL
from PIL import ImageFont, Image, ImageFile, ImageDraw
from matplotlib import pyplot
from torch.utils.data import IterableDataset
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
        super(RecognizerDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.jouyou_kanji

    def generate(self, visualize_label=False):
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


class RecognizerDataset(IterableDataset):
    def __init__(self, fonts_folder: str, background_image_folder: str, transform=None):
        super(RecognizerDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.jouyou_kanji

    def generate(self, visualize_label=False):
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

class HiraganaDataset(IterableDataset):
    def __init__(self, fonts_folder: str, background_image_folder: str, transform=None):
        super(HiraganaDataset).__init__()
        self.font_files = font_paths(fonts_folder)
        self.background_images = background_images(background_image_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = kanji.simple_hiragana

    def generate(self, visualize_label=False):
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

if __name__ == '__main__':
    def generate_grid(dataset, rows=15, columns=5):
        images = [[dataset.generate(True)[0] for i in range(columns)] for j in range(rows)]
        fig, axes = pyplot.subplots(nrows=rows, ncols=columns)
        for axes, images in zip(axes, images):
            for axis, image in zip(axes, images):
                axis.axis('off')
                axis.imshow(image)
        fig.subplots_adjust(0, 0, 1, 1)
        fig.show()

    # generate_grid(RecognizerDataset("data/fonts", "data/background-images"))
    # generate_grid(BoxerDataset("data/fonts", "data/background-images"))
    generate_grid(HiraganaDataset("data/fonts", "data/background-images"))

    dataset = HiraganaDataset("data/fonts", "data/background-images")
    iterator = iter(dataset)
    labels = []
    characters = []
    for i in range(5000):
        sample, label = next(iterator)
        labels += [label]
        characters += [dataset.characters[label]]
        sample.save(f"data/generated/hiragana/{i}.png")
    np.save(f"data/generated/hiragana.npy", np.array(labels, dtype=np.int64))
