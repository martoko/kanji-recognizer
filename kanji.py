from itertools import chain


def characters(ranges):
    return [chr(x) for x in list(chain(*[range(begin, end) for begin, end in ranges]))]


hiragana = characters([[0x3041, 0x3096]])
simple_hiragana = "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどな" \
                  "にぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑを"
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
