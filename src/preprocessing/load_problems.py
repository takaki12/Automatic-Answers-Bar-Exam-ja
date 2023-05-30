# データを読み込んでリストに変換する.
# データ整形もする. (改行やいらない部分の削除など)

import re
import csv
from kansuji_converter import convert_kansuji_to_number
import legal_datalist

# 正規表現
ptn_id = re.compile("<pair id=[\"|\'](.*?)[\"|\']") # 番号番号
ptn_label = re.compile("label=[\"|\'](.*?)[\"|\']") # 問題ラベル
ptn_article_num = re.compile("^第(.*?)条の?([一|二|三|四|五|六|七|八|九]*?)") # 条文番号
pnt_item_num = re.compile("^([１|２|３|４|５|６|７|８|９])") # 項番号

# (第一条 債権は...)を[債権は,1]のように返す、numはString型で返却される点に注意、(例,第二百五十五条の二　→　255-2)
def check_article_num(data,ZyoNum):
    # m1:第X条のY, m2:第X条 のどちらかを抽出する
    m1 = re.match('第[一|二|三|四|五|六|七|八|九|十|百]+条の[一|二|三|四|五|六|七|八|九|十|百]+', data)
    m2 = re.match('第[一|二|三|四|五|六|七|八|九|十|百]+条', data)
    if m1:
        stringobje = re.sub('第', '', m1.group())
        stringobje = re.sub('条の[一|二|三|四|五|六|七|八|九|十|百]+', '', stringobje)
        num = str(convert_kansuji_to_number(stringobje))
        stringobje = re.sub('第.*条の', '', m1.group())
        num = num + '-' + str(convert_kansuji_to_number(stringobje))
    elif m2:
        stringobje = re.sub('第', '', m2.group())
        stringobje = re.sub('条', '', stringobje)
        num = str(convert_kansuji_to_number(stringobje))
    else:
        num = str(ZyoNum)
        
    # 第X条などを除去した文をつくる
    data = re.sub("第[一|二|三|四|五|六|七|八|九|十|百]+条の[一|二|三|四|五|六|七|八|九|十|百]+((\u3000)|( )|(　))","", data)
    data = re.sub("第[一|二|三|四|五|六|七|八|九|十|百]+条((\u3000)|( )|(　))","", data)
    data = re.sub("\u3000","",data)
    data = re.sub("\n","",data)
    return data, num

# 与えられた関連条文の辞書を作成.
def make_minpoudict(texts):
    """
    ex (H18-1-3)
    「第六百九十七条　義務なく他人のために事務の管理を始めた者は、(以下略)
    ２　管理者は、本人の意思を知っているとき、(以下略)」
    が関連条文として与えられたとき、
    {617 : [義務なく他人のために事務の管理を始めた者は、(以下略)], 617-2: [管理者は、本人の意思を知っているとき、(以下略)] }
    上のような辞書を作成する

    データ加工
    1. 「次に掲げる〇〇」の置き換え
    「第百六十六条　債権は、次に掲げる場合には、時効によって消滅する。
    一　債権者が権利を行使することができることを知った時から五年間行使しないとき。
    二　権利を行使することができる時から十年間行使しないとき。
    このような入力に対し、"次に掲げる場合"の"場合"を各号に置き換えたい.」
    {66: [債権は、債権者が権利を行使することができることを知った時から五年間行使しないときには、時効によって消滅する。,債権は、権利を行使することができる時から十年間行使しないときには、時効によって消滅する。]}

    2. 「ただし、~~、この限りでない」の置き換え
    「第九条　成年被後見人の法律行為は、取り消すことができる。ただし、日用品の購入その他日常生活に関する行為については、この限りでない。」
    「ただし、~~、この限りでない。」がある場合、この前後で文を分割する。
    このとき、「ただし、~~、この限りでない。」の部分だけでは意味がわかりにくいため、前の文章をもとに書き換えたものに置き換える.
    (ex)
    ただし、日用品の購入その他日常生活に関する行為については、この限りでない。
    ↓
    成年被後見人の日用品の購入その他日常生活に関する行為については、取り消すことができない。

    Args:
        texts (str): 関連条文 (<t1>にあたる)

    Returns:
        dict: この問題に必要となる条文の辞書
    """

    data_list = []
    for text in texts.split('\n'):
        if text == '':
            continue
        data_list.append(text)

    Minpoudict = {}
    Tuginikakageru_beforetext = ''
    Tuginikakageru_aftertext = ''

    article_num = '0'
    kou_num = '0'
    Kakugou_num = '零'
    condition = 0 # 次に掲げる(各号)に該当する文かどうか (該当で1)
    for data in data_list:
        if condition != 0: # 各号部分
            if re.match('^[一|二|三|四|五|六|七|八|九|十|百]+', data):
                Kakugou_num = re.match('^([一|二|三|四|五|六|七|八|九|十]+)', data).group(1)
                data = data.replace(Kakugou_num, '', 1)
                data = data.replace('。','')
                condition += 1

            elif data.startswith('イ') or data.startswith('ロ') or data.startswith('ハ'):
                # イロハはパス
                pass

            else:
                # 各号おわり
                condition = 0

        befere_article_num = article_num
        data, article_num = check_article_num(data, article_num)
        if befere_article_num != article_num:
            kou_num = "1"
        
        #"２　組合の業務の決定及び執行は",などの大文字の数字を削除
        for smallint,Largenum in legal_datalist.LargeNumDict.items():
                if data.startswith(Largenum):
                    kou_num = str(smallint)
                    data = data[1:]

        # 次に掲げる判定
        if '次に掲げる'  in data or '次の各号に掲げる' in data or '次のとおり' in data:
            # 1行目　Aは次に掲げる権利を持つ
            # 2行目　人権,三行目 投票権　
            # ↑のようになっているため、一行目を読み飛ばし二行目から代入していく
            #Tuginikakageru_beforetextに "Aは" ,Tuginikakageru_aftertextに "を持つ" を代入する

            finednext = False
            for next in legal_datalist.nextlist:
                if ('次に掲げる'+next) in data:
                    Tuginikakageru_beforetext = re.split(('次に掲げる'+next),data)[0]
                    Tuginikakageru_aftertext = re.split(('次に掲げる'+next),data)[1]
                    finednext=True
            if '次の各号に掲げる' in data:
                Tuginikakageru_beforetext = re.split('次の各号に掲げる',data)[0]
                Tuginikakageru_aftertext = re.split('次の各号に掲げる',data)[1]
                finednext=True
            if '次のとおり' in data:
                Tuginikakageru_beforetext = re.split('次のとおり',data)[0]
                Tuginikakageru_aftertext = re.split('次のとおり',data)[1]
                finednext=True
            if not finednext:
                print('new次に掲げる:'+data)
            condition = 1

        if condition == 1:
            # 次に掲げるが含まれている!!
            pass
        elif data.startswith('イ') or data.startswith('ロ') or data.startswith('ハ'):
            # イロハは捨て
            pass
        elif re.search('。',data) or condition != 0:
            # 条文などのとき

            #次に掲げる内の箇条書きの処理
            if condition != 0:
                data = Tuginikakageru_beforetext + data + Tuginikakageru_aftertext

            # ただしの処理
            tadashi_text = ''
            if 'ただし、' in data:
                tadashi_idx = data.index('ただし、')
                tadashi_text = data[tadashi_idx+4:]
                data = data[:tadashi_idx]
            
            if re.search('.*?この限りでない', tadashi_text) and article_num + '_' + kou_num in legal_datalist.minpoudict_withKou_tadashi.keys():
                tadashi_text = legal_datalist.minpoudict_withKou_tadashi[article_num + '_' + kou_num]
            
            # すでにキーがあればappend
            if article_num + "_" + kou_num in Minpoudict:
                Minpoudict[article_num + "_" + kou_num].append(data)
                if tadashi_text != '' and tadashi_text not in Minpoudict[article_num + "_" + kou_num]:
                    Minpoudict[article_num + "_" + kou_num].append(tadashi_text)
            else: # ないなら作る
                Minpoudict[article_num + "_" + kou_num] = [data]
                if tadashi_text != '' and tadashi_text not in Minpoudict[article_num + "_" + kou_num]:
                    Minpoudict[article_num + "_" + kou_num].append(tadashi_text)

        elif re.search('(.*)',data) or re.search('第.章',data) \
        or re.search('第.節',data) or re.search('第.款',data):#()、章、節、款のとき読み飛ばす
            pass
        else:#上記のどれにも当てはまらない時
            print('error20221:'+data)

    return Minpoudict

# 司法試験問題のxmlを読み込んでデータリストに変換する!
def load_problems(year, mode=1):
    '''
    学習・検証・テストに使うxmlファイルを読み込んでリストで出力する.

    Args:
        year (str): 読み込む年度の指定 (drive_dir内のファイルのみ)
        mode (int): 0か1で指定. 0の場合、正解ラベルがなくても問題を読み込む & ラベルを出力しない.

    Returns:
        problems (list[list]): [[label, article(そのまま), problem, number, article_dict], ...]
    '''
    data_dir = "data/row/COLIEE2023statute_data-Japanese/"
    if mode != 0:
        data_dir += 'train'
    else:
        data_dir += 'test'

    coliee_problems = []

    # ファイル名 : riteval_H18_jp.xml. H18のところを変えていく
    file = open(data_dir + "/riteval_" + year + "_jp.xml", 'r')

    number = '' # 問題番号
    label = 0 # 正解ラベル
    article_dict = {}
    article = '' # 条文
    problem = '' # 問題文
    condition = 0 # 1:条文 or 2:問題文 かを判定

    for line in file.readlines():
        line = line.replace('\n','')

        # 読点の統一
        line = re.sub(',|，','、', line)

        # 問題番号、ラベルを読み込む
        if re.search(ptn_id, line):
            match = re.search(ptn_id, line)
            number = match.group(1)
            if number == '':
                print("Number is Unknown")
                print(line)
                exit()

        if re.search(ptn_label, line) and mode != 0:
            match = re.search(ptn_label, line).group(1)
            if match == 'Y':
                label = 1
            elif match == 'N':
                label = 0
            else:
                print("Label is Unknown")
                print(line)
                exit()

        # 関連条文スタート
        if '<t1>' in line:
            condition = 1

        # 関連条文に文章がある場合の処理
        if condition == 1 and (line != '<t1>') and (line != '</t1>'):
            line = line.replace('<t1>','') # タグは消す
            article += line + '\n'
    
        # 関連条文エンド
        if '</t1>' in line:
            condition = 0
            article = article.replace('</t1>','') # タグは消す
            article_dict = make_minpoudict(article)
        
        # 問題文スタート
        if '<t2>' in line:
            condition = 2
            if line == '<t2>':
                continue

        # 特殊処理 (H26・H27用) いらない部分は消す -----------------------------
        if 'に関する次の１から５までの各記述のうち、正しいものはどれか。' in line:
            line = line.replace('に関する次の１から５までの各記述のうち、正しいものはどれか。', '、')
        elif 'ことを前提として、次の１から４までの各記述のうち、誤っているものを２個選びなさい' in line:
            line = line.replace('ことを前提として、次の１から４までの各記述のうち、誤っているものを２個選びなさい', '')
            
        elif re.search('〔第.*?問〕（配点：.*?）', line):
            line = re.sub('〔第.*?問〕（配点：.*?）', '', line)
        # ---------------------------------------------
        
        if condition == 2 and (line != '</t2>'):
            line = line.replace('<t2>','') # タグは消す
            problem += line

        # 問題文エンド
        if '</t2>' in line:
            condition = 0
            problem = problem.replace('</t2>','') # タグは消す

        # 1問分の読み込み終了
        if '</pair>' in line:
            problem = problem.replace('</pair>','') # タグ消し

            if mode != 0:
                coliee_problems.append([label, article, problem, number, article_dict])
            else:
                coliee_problems.append([article, problem, number, article_dict])
            
            # 初期化
            number = ''
            label = ''
            article_dict = {}
            article = ''
            problem = ''
            condition = 0

        file.close()

    return coliee_problems


if __name__=='__main__':
    problems = load_problems('H18')
    for i, problem in enumerate(problems):
        print(problem)