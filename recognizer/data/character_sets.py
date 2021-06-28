import pathlib
from itertools import chain


def characters(ranges):
    return [chr(x) for x in list(chain(*[range(begin, end) for begin, end in ranges]))]


hiragana = characters([[0x3041, 0x3096]])
simple_hiragana = list("あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどな"
                       "にぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑを")
simpler_hiragana = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわを")
aiueo = list("あいうえおが")
katakana_full_width = characters([[0x30A0, 0x30FF]])
kanji = characters([[0x3400, 0x4DB5], [0x4E00, 0x9FCB], [0xF900, 0xFA6A]])
kanji_radicals = characters([[0x2E80, 0x2FD5]])
half_width_katakana_and_punctuation = characters([[0xFF5F, 0xFF9F]])
symbols_and_Punctuation = characters([[0x3000, 0x303F]])
misc_symbols_and_characters = characters([[0x31F0, 0x31FF], [0x3220, 0x3243], [0x3280, 0x337F]])
alphanumeric_and_punctuation = characters([[0xFF01, 0xFF5E]])
all_characters = hiragana + katakana_full_width + kanji + kanji_radicals + \
                 half_width_katakana_and_punctuation + symbols_and_Punctuation + \
                 misc_symbols_and_characters + alphanumeric_and_punctuation

jouyou_kanji = list(pathlib.Path("data/characters/jouyou.txt").read_text().replace("\n", ""))

frequent_kanji = list(pathlib.Path("data/characters/frequent_kanji.txt").read_text().replace("\n", ""))

frequent_kanji_plus = list(pathlib.Path("data/characters/frequent_kanji_plus.txt").read_text().replace("\n", ""))

jouyou_kanji_and_simple_hiragana = jouyou_kanji + simple_hiragana

top_100_kanji = frequent_kanji[:100]

character_sets = {
    "kanji": kanji,  # 27,882
    "jouyou_kanji": jouyou_kanji,  # 1,006
    "top_100_kanji": top_100_kanji,  # 100
    "frequent_kanji": frequent_kanji,  # 2,501
    "frequent_kanji_plus": frequent_kanji_plus,  # 2,502
    "jouyou_kanji_and_simple_hiragana": jouyou_kanji_and_simple_hiragana,  # 1,078
    "simple_hiragana": simple_hiragana,  # 72
    "simpler_hiragana": simpler_hiragana,  # 45
    "aiueo": aiueo  # 6
}

if __name__ == "__main__":
    print("常用漢字")
    print(jouyou_kanji[:10])
    print()

    print("平仮名")
    print(hiragana[:10])
    print()

    print("漢字")
    print(kanji[:10])
    print()

    print("symbols_and_Punctuation")
    print(symbols_and_Punctuation[:20])
    print()
