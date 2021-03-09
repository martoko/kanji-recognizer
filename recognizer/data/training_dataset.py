import glob
import math
import os
import pathlib
import random
from random import randint
from typing import *

import numpy as np
from PIL import ImageFont, Image, ImageFile, ImageDraw
from fontTools.ttLib import TTFont
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from recognizer.data import character_sets

ImageFile.LOAD_TRUNCATED_IMAGES = True


def font_paths(folder):
    return glob.glob(os.path.join(folder, '**/*.ttf'), recursive=True) + \
           glob.glob(os.path.join(folder, '**/*.otf'), recursive=True)


def font_infos(characters, folder):
    def has_glyph(font, glyph):
        for table in font['cmap'].tables:
            if ord(glyph) in table.cmap.keys():
                return True
        return False

    infos = []
    for path in font_paths(folder):
        font = TTFont(path)
        supported_glyphs = set()
        missing_glyphs = set()
        for character in characters:
            if has_glyph(font, character):
                supported_glyphs.add(character)
            else:
                missing_glyphs.add(character)
        if len(missing_glyphs) > 0:
            print(f"{len(missing_glyphs)}/{len(characters)} characters are missing from {os.path.basename(path)}")
        infos += [{
            "path": path,
            "supported_glyphs": supported_glyphs,
            "missing_glyphs": missing_glyphs
        }]
    return infos


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


def random_noise(width, height):
    return Image.fromarray(np.random.randint(0, 255, (width, height, 3), dtype=np.dtype('uint8')))


class RecognizerTrainingDataset(IterableDataset):
    # TODO: Rotate/morph/skew (common one is squishing the characters to fit more on one line)
    # TODO: Other characters as part of background?
    # TODO: Add random lines to image, also in same color as text
    # TODO: Legible text is more common than completely illegible text
    # TODO: Italic
    # TODO: Add random kanji outside of y
    def __init__(self, data_folder: str,
                 character_set: List[str],
                 img_background_weight=1,
                 noise_background_weight=1,
                 plain_background_weight=1, transform=None):
        super().__init__()
        fonts_folder = os.path.join(data_folder, "fonts")
        background_images_folder = os.path.join(data_folder, "backgrounds")
        self.font_infos = font_infos(character_set, fonts_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = character_set
        self.plain_background_weight = plain_background_weight
        self.noise_background_weight = noise_background_weight
        self.img_background_weight = img_background_weight
        self.background_images = [
            Image.open(os.path.join(background_images_folder, name))
            for name in os.listdir(background_images_folder)
            if os.path.isfile(os.path.join(background_images_folder, name))
        ]
        self.stage = 0

    def fonts_supporting_glyph(self, glyph):
        return [info for info in self.font_infos if glyph in info['supported_glyphs']]

    def random_background_image(self, width, height):
        background = random.choice(self.background_images)

        bg_left = randint(0, background.width - 2)
        bg_right = randint(bg_left + 1, background.width)
        bg_top = randint(0, background.height - 2)
        bg_bottom = randint(bg_top + 1, background.height)

        return background.resize(
            (width, height), box=(bg_left, bg_top, bg_right, bg_bottom)
        )

    def generate_background(self, width, height):
        choice = random.choices(["noise", "img", "plain"],
                                weights=[self.noise_background_weight, self.img_background_weight,
                                         self.plain_background_weight])[0]

        if choice == "noise":
            return random_noise(width, height)
        elif choice == "img":
            return self.random_background_image(width, height)
        else:
            return Image.new('RGB', (width, height), color=random_white_color())

    def random_font_size(self):
        return int(random.choices([
            np.random.normal(15, 3),
            np.random.normal(20, 3),
            np.random.normal(35, 3),
            np.random.normal(50, 3)
        ], weights=[10, 3, 1, 1])[0])

    def generate_stage_0(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = self.fonts_supporting_glyph(character)[0]
        font_size = 20
        font = ImageFont.truetype(font_info['path'], font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        sample = Image.new('RGB', (128, 128), color=random_white_color())
        drawing = ImageDraw.Draw(sample)
        drawing.text((64, 64), character, font=font, fill=random_color(), anchor='mm', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate_stage_1(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = self.random_font_size()
        font_size = max(8, font_size)
        font = ImageFont.truetype(font_info['path'], font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        sample = Image.new('RGB', (128, 128), color=random_white_color())
        drawing = ImageDraw.Draw(sample)
        drawing.text((64, 64), character, font=font, fill=random_color(), anchor='mm', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate_stage_2(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = self.random_font_size()
        font_size = max(8, font_size)
        font = ImageFont.truetype(font_info['path'], font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        x_offset = int(((width / 2) - random.random() * width) * 0.8)
        y_offset = int(((height / 2) - random.random() * height) * 0.8)

        sample = Image.new('RGB', (128, 128), color=random_white_color())
        drawing = ImageDraw.Draw(sample)
        drawing.text((64 + x_offset, 64 + y_offset), character, font=font, fill=random_color(), anchor='mm',
                     language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate_stage_3(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = self.random_font_size()
        font_size = max(8, font_size)
        font = ImageFont.truetype(font_info['path'], font_size)

        before_count = random.randint(0, 10)
        after_count = random.randint(0, 10)
        total_count = before_count + after_count + 1
        before = [random.choice(tuple(font_info['supported_glyphs'])) for _ in range(before_count)]
        after = [random.choice(tuple(font_info['supported_glyphs'])) for _ in range(after_count)]
        text = ''.join(before) + character + ''.join(after)

        for character in list(text):
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"'{character}' is missing from {os.path.basename(font_info['path'])}")
                exit(-1)

        left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

        character_width = right / total_count
        character_height = bottom
        x = 64 - character_width / 2 - character_width * before_count
        x_offset = int(((character_width / 2) - random.random() * character_width) * 0.8)
        y_offset = int(((character_height / 2) - random.random() * character_height) * 0.8)

        sample = Image.new('RGB', (128, 128), color=random_white_color())
        drawing = ImageDraw.Draw(sample)
        drawing.text((x + x_offset, 64 + y_offset), text, font=font, fill=random_color(), anchor='lm', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate_stage_4(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = int(random.choices([np.random.normal(15, 3), np.random.normal(20, 3), np.random.normal(35, 3)],
                                       weights=[10, 3, 1])[0])
        font_size = max(8, font_size)
        font = ImageFont.truetype(font_info['path'], font_size)

        before_count = random.randint(0, 10)
        after_count = random.randint(0, 10)
        total_count = before_count + after_count + 1
        before = [random.choice(tuple(font_info['supported_glyphs'])) for _ in range(before_count)]
        after = [random.choice(tuple(font_info['supported_glyphs'])) for _ in range(after_count)]
        text = ''.join(before) + character + ''.join(after)

        for character in list(text):
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"{character} is missing from {os.path.basename(font_info['path'])}")
                exit(-1)

        left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

        character_width = right / total_count
        character_height = bottom
        x = 64 - character_width / 2 - character_width * before_count
        x_offset = int(((character_width / 2) - random.random() * character_width) * 0.8)
        y_offset = int(((character_height / 2) - random.random() * character_height) * 0.8)

        sample = self.generate_background(128, 128)
        drawing = ImageDraw.Draw(sample)
        drawing.text((x + x_offset, 64 + y_offset), text, font=font, fill=random_color(), anchor='lm', language='ja')

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate(self):
        low = math.floor(self.stage)
        high = low + 1
        if random.random() > self.stage - math.floor(self.stage):
            stage = low
        else:
            stage = high

        stage = min(stage, 4)

        if stage == 0:
            return self.generate_stage_0()
        if stage == 1:
            return self.generate_stage_1()
        if stage == 2:
            return self.generate_stage_2()
        if stage == 3:
            return self.generate_stage_3()
        if stage == 4:
            return self.generate_stage_4()

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate()

    def __getitem__(self, index) -> T_co:
        return self.generate()


if __name__ == '__main__':
    files = glob.glob(f"generated/training/*/*")
    for file in files:
        os.remove(file)


    def generate(dataset, stage, count=None):
        dataset.stage = stage
        pathlib.Path(f"generated/training/{stage}").mkdir(parents=True, exist_ok=True)
        iterator = iter(dataset)
        for i in range(len(dataset.characters) if count is None else count):
            sample, label = next(iterator)
            sample.save(f"generated/training/{stage}/{i}.png")


    dataset = RecognizerTrainingDataset(data_folder="data", character_set=character_sets.frequent_kanji_plus)
    generate(dataset, 0, count=200)
    generate(dataset, 1, count=200)
    generate(dataset, 2, count=200)
    generate(dataset, 3, count=200)
    generate(dataset, 4, count=200)
