import argparse
import os
import random
from os import path, listdir
from itertools import chain
from random import randint

from PIL import Image, ImageDraw, ImageFont

# http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-japanese-text/
hiragana = [[0x3041, 0x3096]]
katakana_full_width = [[0x30A0, 0x30FF]]
kanji = [[0x3400, 0x4DB5], [0x4E00, 0x9FCB], [0xF900, 0xFA6A]]
kanji_radicals = [[0x2E80, 0x2FD5]]
half_width_katakana_and_punctuation = [[0xFF5F, 0xFF9F]]
symbols_and_Punctuation = [[0x3000, 0x303F]]
misc_symbols_and_characters = [[0x31F0, 0x31FF], [0x3220, 0x3243], [0x3280, 0x337F]]
alphanumeric_and_punctuation = [[0xFF01, 0xFF5E]]
all_character_ranges = hiragana + katakana_full_width + kanji + kanji_radicals + \
                       half_width_katakana_and_punctuation + symbols_and_Punctuation + \
                       misc_symbols_and_characters + alphanumeric_and_punctuation
all_characters = list(chain.from_iterable([range(begin, end) for begin, end in all_character_ranges]))


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)

def random_white_color():
    return randint(230, 255), randint(230, 255), randint(230, 255)

def random_black_color():
    return randint(0, 25), randint(0, 25), randint(0, 25)


def generate_image(args, character: str, background_image: Image.Image):
    font = ImageFont.truetype(args.font, randint(10, 50))

    if background_image is not None:
        left = randint(0, background_image.width - 64)
        right = left + 64
        top = randint(0, background_image.height - 64)
        bottom = top + 64

        image = background_image.crop((left, top, right, bottom))
    else:
        image = Image.new('RGB', (64, 64), color=random_color())

    drawing = ImageDraw.Draw(image)
    text_color = random.sample([random_color(), random_black_color(), random_white_color()], 1)[0]
    drawing.text((randint(5, 45), randint(5, 35)), character, font=font, fill=text_color, anchor='mm')

    return image


def generate_images(args):
    if not path.exists(args.output_path):
        os.mkdir(args.output_path)

    background_images = []
    if args.background_image_folder is not None:
        background_images = [
            Image.open(path.join(args.background_image_folder, name))
            for name in listdir(args.background_image_folder)
            if path.isfile(path.join(args.background_image_folder, name))
        ]

    for repetition in range(args.repetition_count):
        if not path.exists(path.join(args.output_path, str(repetition))):
            os.mkdir(path.join(args.output_path, str(repetition)))

        for i in range(args.character_count):
            background_image = None
            if len(background_images) > 0:
                background_image = random.sample(background_images, 1)[0]

            image = generate_image(args, chr(all_characters[i % len(all_characters)]), background_image)
            image.save(path.join(args.output_path, str(repetition), f'{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images with kanji on them.')
    parser.add_argument('-c', '--character-count', type=int, default=len(all_characters),
                        help='amount of characters to generate')
    parser.add_argument('-s', '--repetition-count', type=int, default=10,
                        help='amount of sets to generate for each character')
    parser.add_argument('-o', '--output-path', type=str, default='generated',
                        help='path to the folder where generated images are going to be saved')
    parser.add_argument('-f', '--font', type=str, default='/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc',
                        help='path to font to use')
    parser.add_argument('-b', '--background-image-folder', type=str,
                        help='path to a folder containing background images')
    args = parser.parse_args()

    generate_images(args)
