import os
import random
from itertools import chain
from random import randint
from typing import Iterator, Optional

from PIL import ImageFont, Image, ImageFile, ImageDraw
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

ImageFile.LOAD_TRUNCATED_IMAGES = True

hiragana = range(0x3041, 0x3096)


# hiragana = [[0x3041, 0x3096]]
# katakana_full_width = [[0x30A0, 0x30FF]]
# kanji = [[0x3400, 0x4DB5], [0x4E00, 0x9FCB], [0xF900, 0xFA6A]]
# kanji_radicals = [[0x2E80, 0x2FD5]]
# half_width_katakana_and_punctuation = [[0xFF5F, 0xFF9F]]
# symbols_and_Punctuation = [[0x3000, 0x303F]]
# misc_symbols_and_characters = [[0x31F0, 0x31FF], [0x3220, 0x3243], [0x3280, 0x337F]]
# alphanumeric_and_punctuation = [[0xFF01, 0xFF5E]]
# all_character_ranges = hiragana + katakana_full_width + kanji + kanji_radicals + \
#                        half_width_katakana_and_punctuation + symbols_and_Punctuation + \
#                        misc_symbols_and_characters + alphanumeric_and_punctuation
# all_characters = list(chain.from_iterable([range(begin, end) for begin, end in all_character_ranges]))


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def random_white_color():
    return randint(230, 255), randint(230, 255), randint(230, 255)


def random_black_color():
    return randint(0, 25), randint(0, 25), randint(0, 25)


def generate_image(character: str, font_file: str, background_image: Optional[Image.Image], party_mode: bool):
    if party_mode:
        font = ImageFont.truetype(font_file, randint(10, 20))

        # TODO add other kanji on the edges, to simulate being part of sentence

        if background_image is not None:
            left = randint(0, background_image.width - 32)
            right = left + 32
            top = randint(0, background_image.height - 32)
            bottom = top + 32

            image = background_image.crop((left, top, right, bottom))
        else:
            image = Image.new('RGB', (32, 32), color=random_color())

        drawing = ImageDraw.Draw(image)
        text_color = random.sample([random_color(), random_black_color(), random_white_color()], 1)[0]
        # text_color = random.sample([random_black_color(), random_white_color()], 1)[0]
        drawing.text((randint(10, 22), randint(10, 22)), character, font=font, fill=text_color, anchor='mm')
        return image
    else:
        font = ImageFont.truetype(font_file, randint(10, 20))
        image = Image.new('RGB', (32, 32), color=random_white_color())
        drawing = ImageDraw.Draw(image)
        drawing.text((randint(13, 19), randint(13, 19)), character, font=font, fill=random_black_color(), anchor='mm')

        return image


class Kanji(IterableDataset):
    def __init__(self, font_file: str, background_image_folder: str, transform, party_mode: bool = False):
        super(Kanji).__init__()
        self.party_mode = party_mode
        self.font_file = font_file
        self.font_files = [font_file]
        self.background_images = [
            Image.open(os.path.join(background_image_folder, name))
            for name in os.listdir(background_image_folder)
            if os.path.isfile(os.path.join(background_image_folder, name))
        ]
        self.transform = transform
        self.character_index = 0

    def generate_kanji(self):
        character = chr(self.characters()[self.character_index])
        label = self.character_index
        sample = generate_image(character, self.font_file, random.sample(self.background_images, 1)[0], self.party_mode)
        self.character_index = (self.character_index + 1) % len(self.characters())
        return self.transform(sample), label

    def characters(self):
        return hiragana

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate_kanji()

    def __getitem__(self, index) -> T_co:
        return self.generate_kanji()
