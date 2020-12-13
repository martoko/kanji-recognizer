import os
import random
from itertools import chain
from random import randint
from typing import Iterator, Optional

from PIL import ImageFont, Image, ImageFile, ImageDraw
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

ImageFile.LOAD_TRUNCATED_IMAGES = True

# hiragana = range(0x3041, 0x3096)

jouyou_kanji = list(
    "一九七二人入八力十下三千上口土夕大女子小山川五天中六円手文日月木水火犬王正出本右四左玉生田白目石立百年休先名字早気竹糸耳虫"
    "村男町花見貝赤足車学林空金雨青草音校森刀万丸才工弓内午少元今公分切友太引心戸方止毛父牛半市北古台兄冬外広母用矢交会合同回寺"
    "地多光当毎池米羽考肉自色行西来何作体弟図声売形汽社角言谷走近里麦画東京夜直国姉妹岩店明歩知長門昼前南点室後春星海活思科秋茶"
    "計風食首夏弱原家帰時紙書記通馬高強教理細組船週野雪魚鳥黄黒場晴答絵買朝道番間雲園数新楽話遠電鳴歌算語読聞線親頭曜顔丁予化区"
    "反央平申世由氷主仕他代写号去打皮皿礼両曲向州全次安守式死列羊有血住助医君坂局役投対決究豆身返表事育使命味幸始実定岸所放昔板"
    "泳注波油受物具委和者取服苦重乗係品客県屋炭度待急指持拾昭相柱洋畑界発研神秒級美負送追面島勉倍真員宮庫庭旅根酒消流病息荷起速"
    "配院悪商動宿帳族深球祭第笛終習転進都部問章寒暑植温湖港湯登短童等筆着期勝葉落軽運遊開階陽集悲飲歯業感想暗漢福詩路農鉄意様緑"
    "練銀駅鼻横箱談調橋整薬館題士不夫欠氏民史必失包末未以付令加司功札辺印争仲伝共兆各好成灯老衣求束兵位低児冷別努労告囲完改希折"
    "材利臣良芸初果刷卒念例典周協参固官底府径松毒泣治法牧的季英芽単省変信便軍勇型建昨栄浅胃祝紀約要飛候借倉孫案害帯席徒挙梅残殺"
    "浴特笑粉料差脈航訓連郡巣健側停副唱堂康得救械清望産菜票貨敗陸博喜順街散景最量満焼然無給結覚象貯費達隊飯働塩戦極照愛節続置腸"
    "辞試歴察旗漁種管説関静億器賞標熱養課輪選機積録観類験願鏡競議久仏支比可旧永句圧弁布刊犯示再仮件任因団在舌似余判均志条災応序"
    "快技状防武承価舎券制効妻居往性招易枝河版肥述非保厚故政査独祖則逆退迷限師個修俵益能容恩格桜留破素耕財造率貧基婦寄常張術情採"
    "授接断液混現略眼務移経規許設責険備営報富属復提検減測税程絶統証評賀貸貿過勢幹準損禁罪義群墓夢解豊資鉱預飼像境増徳慣態構演精"
    "総綿製複適酸銭銅際雑領導敵暴潔確編賛質興衛燃築輸績講謝織職額識護亡寸己干仁尺片冊収処幼庁穴危后灰吸存宇宅机至否我系卵忘孝困"
    "批私乱垂乳供並刻呼宗宙宝届延忠拡担拝枚沿若看城奏姿宣専巻律映染段洗派皇泉砂紅背肺革蚕値俳党展座従株将班秘純納胸朗討射針降除"
    "陛骨域密捨推探済異盛視窓翌脳著訪訳欲郷郵閉頂就善尊割創勤裁揮敬晩棒痛筋策衆装補詞貴裏傷暖源聖盟絹署腹蒸幕誠賃疑層模穀磁暮誤"
    "誌認閣障劇権潮熟蔵諸誕論遺奮憲操樹激糖縦鋼厳優縮覧簡臨難臓警"
)
kanji = [ord(x) for x in jouyou_kanji]


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
            left = randint(0, background_image.width - 10)
            right = randint(left + 5, background_image.width)
            top = randint(0, background_image.height - 10)
            bottom = randint(top + 5, background_image.height)

            image = background_image.crop((left, top, right, bottom)).resize((32, 32))
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
        return kanji

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.generate_kanji()

    def __getitem__(self, index) -> T_co:
        return self.generate_kanji()
