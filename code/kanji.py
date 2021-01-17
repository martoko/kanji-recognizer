from itertools import chain


def characters(ranges):
    return [chr(x) for x in list(chain(*[range(begin, end) for begin, end in ranges]))]


hiragana = characters([[0x3041, 0x3096]])
simple_hiragana = list("あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどな"
                       "にぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑを")
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

frequent_kanji = list(
    "日一国会人年大十二本中長出三同時政事自行社見月分議後前民生連五発間対上部東者党地合市業内相方四定今回新場金員九入選立開手米"
    "力学問高代明実円関決子動京全目表戦経通外最言氏現理調体化田当八六約主題下首意法不来作性的要用制治度務強気小七成期公持野協取"
    "都和統以機平総加山思家話世受区領多県続進正安設保改数記院女初北午指権心界支第産結百派点教報済書府活原先共得解名交資予川向際"
    "査勝面委告軍文反元重近千考判認画海参売利組知案道信策集在件団別物側任引使求所次水半品昨論計死官増係感特情投示変打男基私各始"
    "島直両朝革価式確村提運終挙果西勢減台広容必応演電歳住争談能無再位置企真流格有疑口過局少放税検藤町常校料沢裁状工建語球営空職"
    "証土与急止送援供可役構木割聞身費付施切由説転食比難防補車優夫研収断井何南石足違消境神番規術護展態導鮮備宅害配副算視条幹独警"
    "宮究育席輸訪楽起万着乗店述残想線率病農州武声質念待試族象銀域助労例衛然早張映限親額監環験追審商葉義伝働形景落欧担好退準賞訴"
    "辺造英被株頭技低毎医復仕去姿味負閣韓渡失移差衆個門写評課末守若脳極種美岡影命含福蔵量望松非撃佐核観察整段横融型白深字答夜製"
    "票況音申様財港識注呼渉達良響阪帰針専推谷古候史天階程満敗管値歌買突兵接請器士光討路悪科攻崎督授催細効図週積丸他及湾録処省旧"
    "室憲太橋歩離岸客風紙激否周師摘材登系批郎母易健黒火戸速存花春飛殺央券赤号単盟座青破編捜竹除完降超責並療従右修捕隊危採織森競"
    "拡故館振給屋介読弁根色友苦就迎走販園具左異歴辞将秋因献厳馬愛幅休維富浜父遺彼般未塁貿講邦舞林装諸夏素亡劇河遣航抗冷模雄適婦"
    "鉄寄益込顔緊類児余禁印逆王返標換久短油妻暴輪占宣背昭廃植熱宿薬伊江清習険頼僚覚吉盛船倍均億途圧芸許皇臨踏駅署抜壊債便伸留罪"
    "停興爆陸玉源儀波創障継筋狙帯延羽努固闘精則葬乱避普散司康測豊洋静善逮婚厚喜齢囲卒迫略承浮惑崩順紀聴脱旅絶級幸岩練押軽倒了庁"
    "博城患締等救執層版老令角絡損房募曲撤裏払削密庭徒措仏績築貨志混載昇池陣我勤為血遅抑幕居染温雑招奈季困星傷永択秀著徴誌庫弾償"
    "刊像功拠香欠更秘拒刑坂刻底賛塚致抱繰服犯尾描布恐寺鈴盤息宇項喪伴遠養懸戻街巨震願絵希越契掲躍棄欲痛触邸依籍汚縮還枚属笑互複"
    "慮郵束仲栄札枠似夕恵板列露沖探逃借緩節需骨射傾届曜遊迷夢巻購揮君燃充雨閉緒跡包駐貢鹿弱却端賃折紹獲郡併草徹飲貴埼衝焦奪雇災"
    "浦暮替析預焼簡譲称肉納樹挑章臓律誘紛貸至宗促慎控贈智握照宙酒俊銭薄堂渋群銃悲秒操携奥診詰託晴撮誕侵括掛謝双孝刺到駆寝透津壁"
    "稲仮暗裂敏鳥純是飯排裕堅訳盗芝綱吸典賀扱顧弘看訟戒祉誉歓勉奏勧騒翌陽閥甲快縄片郷敬揺免既薦隣悩華泉御範隠冬徳皮哲漁杉里釈己"
    "荒貯硬妥威豪熊歯滞微隆埋症暫忠倉昼茶彦肝柱喚沿妙唱祭袋阿索誠忘襲雪筆吹訓懇浴俳童宝柄驚麻封胸娘砂李塩浩誤剤瀬趣陥斎貫仙慰賢"
    "序弟旬腕兼聖旨即洗柳舎偽較覇兆床畑慣詳毛緑尊抵脅祝礼窓柔茂犠旗距雅飾網竜詩昔繁殿濃翼牛茨潟敵魅嫌魚斉液貧敷擁衣肩圏零酸兄罰"
    "怒滅泳礎腐祖幼脚菱荷潮梅泊尽杯僕桜滑孤黄煕炎賠句寿鋼頑甘臣鎖彩摩浅励掃雲掘縦輝蓄軸巡疲稼瞬捨皆砲軟噴沈誇祥牲秩帝宏唆鳴阻泰"
    "賄撲凍堀腹菊絞乳煙縁唯膨矢耐恋塾漏紅慶猛芳懲郊剣腰炭踊幌彰棋丁冊恒眠揚冒之勇曽械倫陳憶怖犬菜耳潜珍梨仁克岳概拘墓黙須偏雰卵"
    "遇湖諮狭喫卓干頂虫刷亀糧梶湯箱簿炉牧殊殖艦溶輩穴奇慢鶴謀暖昌拍朗丈鉱寛覆胞泣涙隔浄匹没暇肺孫貞靖鑑飼陰銘鋭随烈尋渕稿枝丹啓"
    "也丘棟壌漫玄粘悟舗妊塗熟軒旭恩毒騰往豆遂晩狂叫栃岐陛緯培衰艇屈径淡抽披廷錦准暑拝磯奨妹浸剰胆氷繊駒乾虚棒寒孜霊帳悔諭祈惨虐"
    "翻墜沼据肥徐糖搭姉髪忙盾脈滝拾軌俵妨盧粉擦鯨漢糸荘諾雷漂懐勘綿栽才拐笠駄添汗冠斜銅鏡聡浪亜覧詐壇勲魔酬紫湿曙紋卸奮趙欄逸涯"
    "拓眼瓶獄筑尚阜彫咲穏顕巧矛垣召欺釣缶萩粧隻葛脂粛栗愚蒸嘉遭架篠鬼庶肌稚靴菅滋幻煮姫誓耕把践呈疎仰鈍恥剛疾征砕謡嫁謙后嘆俣菌"
    "鎌巣泥頻琴班淵棚潔酷宰廊寂辰隅偶霞伏灯柏辛磨碁俗漠邪晶辻麦墨鎮洞履劣那殴娠奉憂朴亭姓淳荻筒鼻嶋怪粒詞鳩柴偉酔惜穫佳潤悼乏胃"
    "該赴桑桂髄虎盆晋穂壮堤飢傍疫累痴搬畳晃癒桐寸郭机尿凶吐宴鷹賓虜膚陶鐘憾畿猪紘磁弥昆粗訂芽尻庄傘敦騎寧濯循忍磐猫怠如寮祐鵬塔"
    "沸鉛珠凝苗獣哀跳灰匠菓垂蛇澄縫僧幾眺唐亘呉凡憩鄭芦龍媛溝恭刈睡錯伯帽笹穀柿陵霧魂枯弊釧妃舶餓腎窮掌麗綾臭釜悦刃縛暦宜盲粋辱"
    "毅轄猿弦嶌稔窒炊洪摂飽函冗涼桃狩舟貝朱渦紳枢碑鍛刀鼓裸鴨符猶塊旋弓幣膜扇脇腸憎槽鍋慈皿肯樋楊伐駿漬燥糾亮墳坪畜紺慌娯吾椿舌"
    "羅坊峡俸厘峰圭醸蓮弔乙倶汁尼遍堺衡呆薫瓦猟羊窪款閲雀偵喝敢畠胎酵憤豚遮扉硫赦挫挟窃泡瑞又慨紡恨肪扶戯伍忌濁奔斗蘭蒲迅肖鉢朽"
    "殻享秦茅藩沙輔曇媒鶏禅嘱胴粕冨迭挿湘嵐椎灘堰獅姜絹陪剖譜郁悠淑帆暁鷲傑楠笛芥其玲奴誰錠拳翔遷拙侍尺峠篤肇渇榎俺劉幡諏叔雌亨"
    "堪叙酢吟逓痕嶺袖甚喬崔妖琵琶聯蘇闇崇漆岬癖愉寅捉礁乃洲屯樽樺槙薩姻巌淀麹賭擬塀唇睦閑胡幽峻曹哨詠炒屏卑侮鋳抹尉槻隷禍蝶酪茎"
    "汎頃帥梁逝滴汽謎琢箕匿爪芭逗苫鍵襟蛍楢蕉兜寡琉痢庸朋坑姑烏藍僑賊搾奄臼畔遼唄孔橘漱呂桧拷宋嬢苑巽杜渓翁藝廉牙謹瞳湧欣窯褒醜"
    "魏篇升此峯殉煩巴禎枕劾菩堕丼租檜稜牟桟榊錫荏惧倭婿慕廟銚斐罷矯某囚魁薮虹鴻泌於赳漸逢凧鵜庵膳蚊葵厄藻萬禄孟鴈狼嫡呪斬尖翫嶽"
    "尭怨卿串已嚇巳凸暢腫粟燕韻綴埴霜餅魯硝牡箸勅芹杏迦棺儒鳳馨斑蔭焉慧祇摯愁鷺楼彬袴匡眉苅讃尹欽薪湛堆狐褐鴎瀋挺賜嵯雁佃綜繕狛"
    "壷橿栓翠鮎芯蜜播榛凹艶帖伺桶惣股匂鞍蔦玩萱梯雫絆錬湊蜂隼舵渚珂煥衷逐斥稀癌峨嘘旛篭芙詔皐雛娼篆鮫椅惟牌宕喧佑蒋樟耀黛叱櫛渥"
    "挨憧濡槍宵襄妄惇蛋脩笘宍甫酌蚕壕嬉囃蒼餌簗峙粥舘銕鄒蜷暉捧頒只肢箏檀鵠凱彗謄諌樫噂脊牝梓洛醍砦丑笏蕨噺抒嗣隈叶凄汐絢叩嫉朔"
    "蔡膝鍾仇伽夷恣瞑畝抄杭寓麺戴爽裾黎惰坐鍼蛮塙冴旺葦礒咸萌饗歪冥偲壱瑠韮漕杵薔膠允眞蒙蕃呑侯碓茗麓瀕蒔鯉竪弧稽瘤澤溥遥蹴或訃"
    "矩厦冤剥舜侠贅杖蓋畏喉汪猷瑛搜曼附彪撚噛卯桝撫喋但溢闊藏浙彭淘剃揃綺徘巷竿蟹芋袁舩拭茜凌頬厨犀簑皓甦洸毬檄姚蛭婆叢椙轟贋洒"
    "貰儲緋貼諜鯛蓼甕喘怜溜邑鉾倣碧燈諦煎瓜緻哺槌啄穣嗜偕罵酉蹄頚胚牢糞悌吊楕鮭乞倹嗅詫鱒蔑轍醤惚廣藁柚舛縞謳杞鱗繭釘弛狸壬硯蝦"
)

jouyou_kanji_and_simple_hiragana = jouyou_kanji + simple_hiragana

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
