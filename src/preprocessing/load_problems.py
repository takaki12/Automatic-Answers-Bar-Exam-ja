# データを読み込んでリストに変換する.
# データ整形もする. (改行やいらない部分の削除など)

import re
import csv

# 正規表現
ptn_id = re.compile("<pair id=[\"|\'](.*?)[\"|\']") # 番号番号
ptn_label = re.compile("label=[\"|\'](.*?)[\"|\']") # 問題ラベル
ptn_article_num = re.compile("^第(.*?)条の?([一|二|三|四|五|六|七|八|九]*?)") # 条文番号
pnt_item_num = re.compile("^([１|２|３|４|５|６|７|８|９])") # 項番号

def load_problems(years, mode=1):
    '''
    学習・検証・テストに使うxmlファイルを読み込んでリストで出力する.

    Args:
        years (list[str]): 読み込む年度の指定 (drive_dir内のファイルのみ)
        mode (int): 0か1で指定. 0の場合、正解ラベルがなくても問題を読み込む & ラベルを出力しない.

    Returns:
        problems (list[list]): [[label, article_list, article(そのまま), problem, number], ...]
    '''
    data_dir = "data/row/COLIEE2023statute_data-Japanese/"
    if mode != 0:
        data_dir += 'train'
    else:
        data_dir += 'test'

    coliee_problems = []
    for year in years:
        # ファイル名 : riteval_H18_jp.xml. H18のところを変えていく
        file = open(data_dir + "/riteval_" + year + "_jp.xml", 'r')

        number = '' # 問題番号
        label = 0 # 正解ラベル
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
                article += line
                    
            # 関連条文エンド
            if '</t1>' in line:
                condition = 0
                article = article.replace('</t1>','') # タグは消す

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
                    coliee_problems.append([label, article, problem, number])
                else:
                    coliee_problems.append([article, problem, number])
                
                # 初期化
                number = ''
                label = ''
                article = ''
                problem = ''
                condition = 0

        file.close()

    return coliee_problems


if __name__=='__main__':
    years = ['H18','H19']
    problems = load_problems(years)
    for i, problem in enumerate(problems):
        if i == 3:
            break
        print(problem)