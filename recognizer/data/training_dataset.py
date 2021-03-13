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

from recognizer.data import character_sets, fonts

ImageFile.LOAD_TRUNCATED_IMAGES = True


def background_images(folder):
    return [
        Image.open(os.path.join(folder, name))
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    ]


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def random_noise(width, height):
    return Image.fromarray(np.random.randint(0, 255, (width, height, 3), dtype=np.dtype('uint8')))


def draw_outlined_text(drawing, xy, *args, outline_fill=None, outline_thickness=None, fill, **kwargs):
    if outline_fill is None:
        outline_fill = random_color()
    if outline_thickness is None:
        outline_thickness = 1 + round(abs(np.random.normal(1)))

    x, y = xy
    for dx in range(-outline_thickness, 1 + outline_thickness):
        for dy in range(-outline_thickness, 1 + outline_thickness):
            drawing.text((x + dx, y + dy), *args, fill=outline_fill, **kwargs)

    drawing.text(xy, *args, fill=fill, **kwargs)


def draw_underlined_text(drawing, xy, text, *args, font, anchor, language, **kwargs):
    left, top, right, bottom = font.getbbox(text, anchor=anchor, language=language)

    width = round(abs(np.random.normal(1, 0.1)))
    jitter = np.random.normal(1, 0.1) * (font.size / 20)
    drawing.text(xy, text, *args, font=font, anchor=anchor, language=language, **kwargs)
    x, y = xy
    drawing.line((
        (x + left, y + bottom + jitter),
        (x + right, y + bottom + jitter)
    ), *args, width=width, **kwargs)


def eat_sides(image, left, right, top, bottom):
    left = round(left)
    right = round(right)
    top = round(top)
    bottom = round(bottom)
    color = random_color()

    drawing = ImageDraw.Draw(image)
    drawing.rectangle((
        (0, 0),
        (random.randint(0, left), image.height)
    ), fill=color)

    drawing.rectangle((
        (image.width, 0),
        (random.randint(right, image.width), image.height)
    ), fill=color)

    drawing.rectangle((
        (0, 0),
        (image.width, random.randint(0, top))
    ), fill=color)

    drawing.rectangle((
        (0, image.height),
        (image.width, random.randint(bottom, image.height))
    ), fill=color)


class RecognizerTrainingDataset(IterableDataset):
    def __init__(self, data_folder: str,
                 character_set: List[str], transform=None):
        super().__init__()
        fonts_folder = os.path.join(data_folder, "fonts")
        background_images_folder = os.path.join(data_folder, "backgrounds")
        self.font_infos = fonts.font_infos_in_folder(fonts_folder, character_set)
        self.transform = transform
        self.character_index = 0
        self.characters = character_set
        self.background_images = [
            Image.open(os.path.join(background_images_folder, name))
            for name in os.listdir(background_images_folder)
            if os.path.isfile(os.path.join(background_images_folder, name))
        ]
        self.stage = 0

    def fonts_supporting_glyph(self, glyph):
        return [font for font in self.font_infos if glyph in font.supported_glyphs]

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
        choice = random.choices(["noise", "img", "plain"])[0]

        if choice == "noise":
            if random.random() > 0.5:
                image = self.random_background_image(width, height)
            else:
                image = Image.new('RGB', (width, height), color=random_color())
            return Image.blend(image, random_noise(width, height), min(abs(np.random.normal(0, 0.3)), 1))
        elif choice == "img":
            return self.random_background_image(width, height)
        else:
            return Image.new('RGB', (width, height), color=random_color())

    @staticmethod
    def random_font_size():
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
        font = font_info.get(font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        sample = Image.new('RGB', (128, 128), color=random_color())
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
        font = font_info.get(font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        sample = Image.new('RGB', (128, 128), color=random_color())
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
        font = font_info.get(font_size)

        _, _, width, height = font.getbbox(character, anchor='lt', language='ja')
        x_offset = int(((width / 2) - random.random() * width) * 0.8)
        y_offset = int(((height / 2) - random.random() * height) * 0.8)

        sample = Image.new('RGB', (128, 128), color=random_color())
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
        font = font_info.get(font_size)

        before_count = random.randint(0, 10)
        after_count = random.randint(0, 10)
        total_count = before_count + after_count + 1
        before = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(before_count)]
        after = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(after_count)]
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

        sample = Image.new('RGB', (128, 128), color=random_color())
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
        font_size = self.random_font_size()
        font_size = max(8, font_size)
        font = font_info.get(font_size)

        before_count = random.randint(0, 10)
        after_count = random.randint(0, 10)
        total_count = before_count + after_count + 1
        before = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(before_count)]
        after = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(after_count)]
        text = ''.join(before) + character + ''.join(after)

        for character in list(text):
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"{character} is missing from {os.path.basename(font_info.path)}")
                exit(-1)

        left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

        character_width = right / total_count
        character_height = bottom
        x = 64 - character_width / 2 - character_width * before_count
        x_offset = int(((character_width / 2) - random.random() * character_width) * 0.8)
        y_offset = int(((character_height / 2) - random.random() * character_height) * 0.8)

        sample = self.generate_background(128, 128)
        drawing = ImageDraw.Draw(sample)

        if random.random() > 0.9:
            draw_outlined_text(drawing, (x + x_offset, 64 + y_offset),
                               text, font=font, fill=random_color(), anchor='lm', language='ja')
        else:
            drawing.text((x + x_offset, 64 + y_offset),
                         text, font=font, fill=random_color(), anchor='lm', language='ja')

        if random.random() > 0.9:
            eat_sides(
                sample,
                64 + x_offset - character_width / 2,
                64 + x_offset + character_width / 2,
                64 + y_offset - character_height / 2,
                64 + y_offset + character_height / 2
            )

        self.character_index = (self.character_index + 1) % len(self.characters)
        if self.transform is None:
            return sample, label
        else:
            return self.transform(sample), label

    def generate_stage_5(self):
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = self.random_font_size()
        font_size = max(8, font_size)
        font = font_info.get(font_size)

        before_count = random.randint(0, 10)
        after_count = random.randint(0, 10)
        total_count = before_count + after_count + 1
        before = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(before_count)]
        after = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(after_count)]
        text = ''.join(before) + character + ''.join(after)

        floating_count = int(abs(np.random.normal(0, 10)))
        floating_characters = [random.choice(tuple(font_info.supported_glyphs)) for _ in range(floating_count)]

        for character in list(text) + floating_characters:
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"{character} is missing from {os.path.basename(font_info.path)}")
                exit(-1)

        left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

        character_width = right / total_count
        character_height = bottom
        x = 64 - character_width / 2 - character_width * before_count
        x_offset = int(((character_width / 2) - random.random() * character_width) * 0.8)
        y_offset = int(((character_height / 2) - random.random() * character_height) * 0.8)
        x = x + x_offset
        y = 64 + y_offset

        sample = self.generate_background(128, 128)
        drawing = ImageDraw.Draw(sample)

        effect = random.choices(['outline', 'underline', 'none'], weights=[1, 1, 10])[0]
        if effect == 'outline':
            draw_outlined_text(drawing, (x, y), text, font=font, fill=random_color(), anchor='lm', language='ja')
        elif effect == 'underline':
            draw_underlined_text(drawing, (x, y), text, font=font, fill=random_color(), anchor='lm', language='ja')
        else:
            drawing.text((x, y), text, font=font, fill=random_color(), anchor='lm', language='ja')

        for character in floating_characters:
            font_info = random.choice(self.fonts_supporting_glyph(character))
            font_size = self.random_font_size()
            font_size = max(8, font_size)
            font = font_info.get(font_size)
            f_left, f_top, f_right, f_bottom = font.getbbox(character, anchor='lt', language='ja')

            floating_x = []
            floating_y = []
            if y - bottom / 2 - f_bottom > -f_bottom:
                floating_y += [random.randint(-f_bottom, y - bottom // 2 - f_bottom)]
            if 128 > y + bottom / 2:
                floating_y += [random.randint(y + bottom // 2, 128)]
            if not floating_y:
                continue

            floating_x += [random.randint(-f_right, 128)]

            if random.random() > 0.9:
                draw_outlined_text(drawing, (random.choice(floating_x), random.choice(floating_y)), character,
                                   font=font,
                                   fill=random_color(), anchor='lt', language='ja')
            else:
                drawing.text((random.choice(floating_x), random.choice(floating_y)), character, font=font,
                             fill=random_color(), anchor='lt', language='ja')

        if random.random() > 0.9:
            eat_sides(
                sample,
                64 + x_offset - character_width / 2,
                64 + x_offset + character_width / 2,
                64 + y_offset - character_height / 2,
                64 + y_offset + character_height / 2
            )

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

        stage = min(stage, 5)

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
        if stage == 5:
            return self.generate_stage_5()

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
    generate(dataset, 0, count=50)
    generate(dataset, 1, count=50)
    generate(dataset, 2, count=50)
    generate(dataset, 3, count=50)
    generate(dataset, 4, count=50)
    generate(dataset, 5, count=200)
