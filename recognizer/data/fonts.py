import glob
import os
from typing import List, Set

from PIL import ImageFont
from fontTools.ttLib import TTFont
import json
from pathlib import Path


def font_paths_in_folder(folder):
    return glob.glob(os.path.join(folder, '**/*.ttf'), recursive=True) + \
           glob.glob(os.path.join(folder, '**/*.otf'), recursive=True)


def font_infos_in_folder(folder, characters):
    return [FontInfo.from_font_cached(path, characters) for path in font_paths_in_folder(folder)]


class FontInfo:
    def __init__(self, path: str, characters: List[str], supported_glyphs: Set[str], missing_glyphs: Set[str]):
        self.path = path
        self.characters = characters
        self.supported_glyphs = supported_glyphs
        self.missing_glyphs = missing_glyphs

    def get(self, size):
        return ImageFont.truetype(self.path, size)

    def to_json(self):
        with open(Path(self.path).with_suffix(".json"), 'w') as file:
            json.dump({
                "path": self.path,
                "characters": self.characters,
                "supported_glyphs": list(self.supported_glyphs),
                "missing_glyphs": list(self.missing_glyphs)
            }, file)

    @staticmethod
    def from_font(path, characters):
        info = TTFont(path)

        def has_glyph(glyph):
            for table in info['cmap'].tables:
                if ord(glyph) in table.cmap.keys():
                    return True
            return False

        supported_glyphs = set()
        missing_glyphs = set()
        for character in characters:
            if has_glyph(character):
                supported_glyphs.add(character)
            else:
                missing_glyphs.add(character)
        if len(missing_glyphs) > 0:
            print(f"{len(missing_glyphs)}/{len(characters)} characters are missing from {os.path.basename(path)}")
        return FontInfo(path, characters, supported_glyphs, missing_glyphs)

    @staticmethod
    def from_json(path):
        with open(Path(path).with_suffix(".json"), 'r') as file:
            data = json.load(file)
            return FontInfo(
                data['path'], data['characters'], set(data['supported_glyphs']), set(data['missing_glyphs'])
            )

    @staticmethod
    def from_font_cached(path, characters):
        if os.path.exists(Path(path).with_suffix(".json")):
            font_info = FontInfo.from_json(Path(path).with_suffix(".json"))
            if font_info.characters != characters:
                font_info = FontInfo.from_font(path, characters)
                font_info.to_json()
            return font_info
        else:
            font_info = FontInfo.from_font(path, characters)
            font_info.to_json()
            return font_info
