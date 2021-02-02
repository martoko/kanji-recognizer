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
from torchvision import transforms

import kanji
from fontTools.ttLib import TTFont

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


# TODO: Go through each font and check what chars are available and build a map, so that we can be smart about it at generation time and so we can provide a single line for each font stating how many chars are missing


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


class KanjiBoxerGeneratedDataset(IterableDataset):
    # TODO: Rotate/morph/skew
    # TODO: Other characters as part of background?
    # TODO: Add a small random padding/cropping to be more resistant to faulty cropping
    # TODO: Add random lines to image, also in same color as text
    # TODO: Legible text is more common than completely illegible text
    # TODO: Italic
    def __init__(self, fonts_folder: str,
                 background_image_folder: str,
                 img_background_weight=1,
                 noise_background_weight=1,
                 plain_background_weight=1,
                 border_ratio=0.5,
                 characters=kanji.jouyou_kanji, transform=None):
        super(KanjiRecognizerGeneratedDataset).__init__()
        self.font_infos = font_infos(characters, fonts_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = characters
        self.plain_background_weight = plain_background_weight
        self.noise_background_weight = noise_background_weight
        self.img_background_weight = img_background_weight
        self.border_ratio = border_ratio
        self.background_images = [
            Image.open(os.path.join(background_image_folder, name))
            for name in os.listdir(background_image_folder)
            if os.path.isfile(os.path.join(background_image_folder, name))
        ]
        self.id = 'boxer-2'

    def fonts_supporting_glyph(self, glyph):
        return [info for info in self.font_infos if glyph in info['supported_glyphs']]

    def random_noise(self, width, height):
        return Image.fromarray(np.random.randint(0, 255, (width, height, 3), dtype=np.dtype('uint8')))

    def background_image(self, width, height):
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
            return self.random_noise(width, height)
        elif choice == "img":
            return self.background_image(width, height)
        else:
            return Image.new('RGB', (width, height), color=random_white_color())

    def generate(self):
        # Perfectly cropped B&W images
        character = self.characters[self.character_index]
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = randint(8, 32)
        font = ImageFont.truetype(font_info['path'], font_size)

        before = random.choice(tuple(font_info['supported_glyphs']))
        after = random.choice(tuple(font_info['supported_glyphs']))
        text = before + character + after

        for character in list(text):
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"{character} is missing from {os.path.basename(font_info['path'])}")
                exit(-1)

        left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

        sample = self.generate_background(32, 32)
        drawing = ImageDraw.Draw(sample)
        x_offset = randint(-int((right / 3) / 2), int((right / 3) / 2))
        y_offset = randint(-int(bottom / 2), int(bottom / 2))
        drawing.text((16 + x_offset, 16 + y_offset), text, font=font,
                     fill=random_color(), anchor='mm',
                     language='ja')
        label = [
            (16 + x_offset - right / 3 / 2),
            (16 + y_offset - bottom / 2),
            (16 + x_offset + right / 3 / 2),
            (16 + y_offset + bottom / 2)
        ]
        drawing = ImageDraw.Draw(sample)
        # drawing.rectangle(label, outline=(255, 0, 0))
        # drawing.point((16, 16), fill=(255, 0, 0))

        if label[0] < 0: label[0] = 0
        if label[1] < 0: label[1] = 0
        if label[2] > 31: label[0] = 31
        if label[3] > 31: label[1] = 31

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
            (32, 32), box=(bg_left, bg_top, bg_right, bg_bottom)
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


class KanjiRecognizerGeneratedDataset(IterableDataset):
    # TODO: Rotate/morph/skew
    # TODO: Other characters as part of background?
    # TODO: Add a small random padding/cropping to be more resistant to faulty cropping
    # TODO: Add random lines to image, also in same color as text
    # TODO: Legible text is more common than completely illegible text
    # TODO: Italic
    def __init__(self, fonts_folder: str,
                 background_image_folder: str,
                 side_text_ratio=0.5,
                 img_background_weight=1,
                 noise_background_weight=1,
                 plain_background_weight=1,
                 border_ratio=0.5,
                 characters=kanji.jouyou_kanji, transform=None):
        super(KanjiRecognizerGeneratedDataset).__init__()
        self.font_infos = font_infos(characters, fonts_folder)
        self.transform = transform
        self.character_index = 0
        self.characters = characters
        self.side_text_ratio = side_text_ratio
        self.plain_background_weight = plain_background_weight
        self.noise_background_weight = noise_background_weight
        self.img_background_weight = img_background_weight
        self.border_ratio = border_ratio
        self.background_images = [
            Image.open(os.path.join(background_image_folder, name))
            for name in os.listdir(background_image_folder)
            if os.path.isfile(os.path.join(background_image_folder, name))
        ]
        self.padding = 3
        self.id = 'recognizer-3'

    def fonts_supporting_glyph(self, glyph):
        return [info for info in self.font_infos if glyph in info['supported_glyphs']]

    def random_noise(self, width, height):
        return Image.fromarray(np.random.randint(0, 255, (width, height, 3), dtype=np.dtype('uint8')))

    def background_image(self, width, height):
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
            return self.random_noise(width, height)
        elif choice == "img":
            return self.background_image(width, height)
        else:
            return Image.new('RGB', (width, height), color=random_white_color())

    def generate(self):
        # Perfectly cropped B&W images
        character = self.characters[self.character_index]
        label = self.character_index
        font_info = random.choice(self.fonts_supporting_glyph(character))
        font_size = randint(12, 22)
        font = ImageFont.truetype(font_info['path'], font_size)

        if random.random() < self.side_text_ratio:
            before = random.choice(tuple(font_info['supported_glyphs']))
            after = random.choice(tuple(font_info['supported_glyphs']))
            text = before + character + after

            for character in list(text):
                left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
                if right == 0 or bottom == 0:
                    print(f"{character} is missing from {os.path.basename(font_info['path'])}")
                    exit(-1)

            left, top, right, bottom = font.getbbox(text, anchor='lt', language='ja')

            sample = self.generate_background(round(right / 3) + 2 * self.padding, bottom + 2 * self.padding)
            drawing = ImageDraw.Draw(sample)
            drawing.text((self.padding - right / 3, self.padding), text, font=font, fill=random_color(), anchor='lt',
                         language='ja')
            sample = sample.resize((32 + 2 * self.padding, 32 + 2 * self.padding))
        else:
            left, top, right, bottom = font.getbbox(character, anchor='lt', language='ja')
            if right == 0 or bottom == 0:
                print(f"{character} is missing from {os.path.basename(font['path'])}")
                exit(-1)

            sample = self.generate_background(right + 2 * self.padding, bottom + 2 * self.padding)
            drawing = ImageDraw.Draw(sample)

            if random.random() < self.border_ratio:
                border_color = random.choice([random_white_color(), random_color()])

                # thin border
                drawing.text((self.padding - 1, self.padding), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding + 1, self.padding), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding, self.padding - 1), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding, self.padding + 1), character, anchor='lt', font=font, fill=border_color)

                # thicker border
                drawing.text((self.padding - 1, self.padding - 1), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding + 1, self.padding - 1), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding - 1, self.padding + 1), character, anchor='lt', font=font, fill=border_color)
                drawing.text((self.padding + 1, self.padding + 1), character, anchor='lt', font=font, fill=border_color)

            drawing.text((self.padding, self.padding), character, font=font, fill=random_color(), anchor='lt',
                         language='ja')
            sample = sample.resize((32 + 2 * self.padding, 32 + 2 * self.padding))

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
    def __init__(self, path, characters=kanji.jouyou_kanji, transform=None):
        super(RecognizerTestDataset).__init__()
        self.id = 'recognizer-test-1'
        self.characters = characters
        self.transform = transform
        sample_paths = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
        label_characters = [os.path.basename(os.path.dirname(path)) for path in sample_paths]
        self.samples = [(path, character) for path, character
                        in zip(sample_paths, label_characters)
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
    def generate(dataset, count=None):
        files = glob.glob(f"data/generated/{dataset.id}/*")
        for file in files:
            os.remove(file)
        pathlib.Path(f"data/generated/{dataset.id}").mkdir(parents=True, exist_ok=True)
        iterator = iter(dataset)
        for i in range(len(dataset.characters) if count is None else count):
            sample, label = next(iterator)
            sample.save(f"data/generated/{dataset.id}/{i}.png")


    def normalization_data(dataset, name, count=4000):
        pixels = np.array([])
        iterator = iter(dataset)
        for i in range(count):
            sample, label = next(iterator)
            pixels = np.append(pixels, np.array(sample) / 255)
            if i % (round(count / 100)) == 0:
                print(f"{round(i / count * 100)}% {i}/{count}\r", end='')
        print(f"{name}: mean {pixels.mean()}, std: {pixels.std()}")


    adversarial_transform = transforms.Compose([
        transforms.RandomCrop((32, 32)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Lambda(lambda img: PIL.ImageOps.invert(img) if random.random() > 0.5 else img),
    ])

    plain_transform = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.Resize((32, 32))
    ])

    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: adversarial_transform(img) if random.random() > 0.1 else plain_transform(img)),
    ])

    # normalization_data(KanjiRecognizerGeneratedDataset("data/fonts", "data/background-images", characters=kanji.frequent_kanji_plus), "recognizer")
    # normalization_data(BoxerDataset("data/fonts", "data/background-images"), "boxer")
    # normalization_data(HiraganaDataset("data/fonts", "data/background-images"), "hiragana")

    # generate(
    #     KanjiRecognizerGeneratedDataset("data/fonts", "data/background-images", characters=kanji.frequent_kanji_plus,
    #                                     transform=plain_transform), count=100)
    generate(
        KanjiBoxerGeneratedDataset("data/fonts", "data/background-images", characters=kanji.frequent_kanji_plus,
                                   transform=None), count=100)
    # generate(BoxerDataset("data/fonts", "data/background-images"), "boxer")
    # generate(HiraganaDataset("data/fonts", "data/background-images"), "hiragana")
