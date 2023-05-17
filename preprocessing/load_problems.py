import os
import re
import csv
import pickle
from tqdm import tqdm
import LegalDataList
import LegalMethod
import data_augumentation
import extract_svo
import itertools
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('studio-ousia/luke-japanese-base-lite')
import sentence_bert

# 正規表現
ptn_id = re.compile("<pair id=[\"|\'](.*?)[\"|\']") # 番号番号
ptn_label = re.compile("label=[\"|\'](.*?)[\"|\']") # 問題ラベル
ptn_article_num = re.compile("^第(.*?)条の?([一|二|三|四|五|六|七|八|九]*?)") # 条文番号
pnt_item_num = re.compile("^([１|２|３|４|５|６|７|８|９])") # 項番号

def main(years, mode='train', topk=5, delete_predicate_bool=True, logical_inversion_bool=True, person_replacement_bool=True):
    problems = load_problems(years, mode)
    selected_problems = []
    for problem in tqdm(problems):
        label = int(problem[0])
        article_list = problem[1]
        article_text = problem[2]
        t2_text = problem[3]
        if mode == 'test':
            number = problem[4]

        # 条文選択
        article_list = slect_article(article_list, t2_text, topk, delete_predicate_bool)
        for a in article_list:
            if mode != 'test':
                selected_problems.append([label, a[0], t2_text])
            else:
                selected_problems.append([label, a[0], t2_text, number])

    # データ拡張
    retrun_datalist = data_augumentaion(selected_problems, logical_inversion_bool, person_replacement_bool)
    return retrun_datalist

# 年別に問題文を読み込み、[番号、ラベル、条文リスト、条文、問題文]のリストを作成
def load_problems(years, mode='train'):
    '''
    colieeのxmlファイルを読み込んでリストで出力する
    データ拡張などの他処理はしない

    Args:
        years (list[str]): 読み込む年度の指定 (drive_dir内のファイルのみ)

    Returns:
        coliee_problems (list[list]): [[label, article_list, article(そのまま), problem, number], ...]
    '''
    drive_dir = "data/raw/COLIEE2023statute_data-Japanese/"
    if mode == 'train' or mode == 'val':
        drive_dir = drive_dir + 'train'
    elif mode == 'test':
        drive_dir = drive_dir + 'test'

    coliee_problems = []
    for year in years:
        file = open(drive_dir + "/riteval_" + year + "_jp.xml", 'r')

        number = ''
        label = 0
        article_num = ''
        article_list = []
        article = ''
        problem = ''
        condition = 0 # 1:条文or2:問題文かを判定

        for line in file.readlines():
            line = line.replace('\n','')

            # 読点の統一
            line = re.sub(',|，','、', line)

            # 問題番号、ラベル
            if re.search(ptn_id, line):
                match = re.search(ptn_id, line)
                number = match.group(1)
                if number == '':
                    print("Number is Unknown")
                    print(line)
                    exit()

            if re.search(ptn_label, line) and mode != 'test':
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

            if condition == 1 and (line != '<t1>') and (line != '</t1>'):
                line = line.replace('<t1>','')
                article_text, article_num, item_num = LegalMethod.check_article_item_number(line, article_num)
                article += line

                if item_num != '':
                    article_item_num = article_num + '_' + item_num
                else:
                    article_item_num = article_num
                if article_item_num != '' and article_item_num not in article_list:
                    article_list.append(article_item_num)
                    
            # 関連条文エンド
            if '</t1>' in line:
                condition = 0
                article = article.replace('</t1>','')

            # 問題文スタート
            if '<t2>' in line:
                condition = 2
                if line == '<t2>':
                    continue

            # 特殊処理 (H26・H27用) -----------------------------
            if 'に関する次の１から５までの各記述のうち、正しいものはどれか。' in line:
                line = line.replace('に関する次の１から５までの各記述のうち、正しいものはどれか。', '、')
            elif 'ことを前提として、次の１から４までの各記述のうち、誤っているものを２個選びなさい' in line:
                line = line.replace('ことを前提として、次の１から４までの各記述のうち、誤っているものを２個選びなさい', '')
                
            elif re.search('〔第.*?問〕（配点：.*?）', line):
                line = re.sub('〔第.*?問〕（配点：.*?）', '', line)
            # ---------------------------------------------
            
            if condition == 2 and (line != '</t2>'):
                line = line.replace('<t2>','')
                problem += line

            # 問題文エンド
            if '</t2>' in line:
                condition = 0
                problem = problem.replace('</t2>','')

            if '</pair>' in line:
                problem = problem.replace('</pair>','')

                if mode == 'train':
                    coliee_problems.append([label, article_list, article, problem, number])
                elif mode == 'test':
                    coliee_problems.append([article_list, article, problem, number])
                
                # 初期化
                number = ''
                label = ''
                article_num = ''
                article_list = []
                article = ''
                problem = ''
                condition = 0

        file.close()

    return coliee_problems

# 問題文から述語を抽出し、消す。
def delete_predicate(t2):
    svo_dict, svo_arrow_text = extract_svo.return_svo_triple(t2)
    predicate = svo_dict['述語']
    t2_no_predicate = t2[:t2.rfind(predicate)]
    return t2_no_predicate

# 必要な条文を全てのパターンから選ぶ。
def slect_article(article_list, t2, topk=5, delete_predicate_bool=False):
    """条文を選択する。上位5件を返します。

    Args:
        article_list (list): 使われている条文番号のリスト
        t2 (_type_): 問題文
        Delete_predicate (bool): 条文選択の際に、問題文の述語を消すかどうか Defaults to False.

    Returns:
        list: 上位topk個のクエリと似た条文リスト
    """

    t1_list = [] # 条文そのままの組合せ
    t1_list_okikae = [] # 参照条文が置き換えられた条文の組合せ
    for k in article_list:
        for i,minpou in enumerate(LegalDataList.Minpoudict_withKou[k]):
            if minpou not in t1_list:
                t1_list.append([i,minpou])

        for i, minpou_okikae in enumerate(LegalDataList.Minpoudict_withKou_Okikae[k]):
            if minpou_okikae not in t1_list_okikae:
                t1_list_okikae.append([i,minpou_okikae])

    sentences = [] # 全組合せ
    max_combination_num = 1000 # 最大組合せ数
    t2_tokens = len(tokenizer.tokenize(t2))

    ptn = itertools.product([0,1], repeat=len(t1_list))
    ptn_t1 = []
    i = 0
    for p in ptn:
        if list(p).count(0) == len(t1_list): # 空っぽは飛ばす
            continue

        if list(p).count(1) == 1: # 1個だけのものは優先して作っておきたい
            ptn_t1.insert(i, p)
            i += 1
        else:
            ptn_t1.append(p)

    # 条文そのままの組合せ(t1_list)の追加
    for bits in ptn_t1:

        if len(sentences) >= max_combination_num:
            break

        tmp_comb = []
        sentence = ''
        for x, article in zip(bits, t1_list):
            if x == 1 and article[0] not in tmp_comb:
                tmp_comb.append(article[0])
                sentence += article[1]
        
        if len(tokenizer.tokenize(sentence)) <= 509-t2_tokens and sentence not in sentences:
            sentences.append(sentence)

    # 参照条文を置き換えた条文の処理
    ptn = itertools.product([0,1], repeat=len(t1_list_okikae))
    ptn_t1_okikae = []
    i = 0
    for p in ptn:
        if list(p).count(0) == len(t1_list_okikae): # 空っぽは飛ばす
            continue

        if list(p).count(1) == 1: # 1個だけのものは優先して作っておきたい
            ptn_t1_okikae.insert(i, p)
            i += 1
        else:
            ptn_t1_okikae.append(p)

    # 参照条文が置き換えられた条文の組み合わせの追加
    for bits in ptn_t1_okikae:
        if len(sentences) >= max_combination_num:
            break

        tmp_comb = []
        sentence = ''
        for x, article in zip(bits, t1_list_okikae):
            if x == 1 and article[0] not in tmp_comb:
                tmp_comb.append(article[0])
                sentence += article[1]
        
        # トークンオーバーとすでに作られている組合せは使わない。
        if len(tokenizer.tokenize(sentence)) <= 509-t2_tokens and sentence not in sentences:
            sentences.append(sentence)

    if delete_predicate_bool:
        query = [delete_predicate(t2)]
    else:
        query = [t2]

    top_list = sentence_bert.compare_sentence(sentences, query, model_type='luke', closest_n=topk)
    return top_list

def data_augumentaion(problem_list, logical_inversion_bool=True, person_replacement_bool=True):
    """問題リストからデータ拡張する

    Args:
        problem_list (list): 問題リスト [label, t1, t2, number]
        logical_inversion_bool (bool): 論理反転する?. Defaults to True.
        person_replacement_bool (bool): 人物名置き換え. Defaults to True.

    Returns:
        list: 問題リスト
    """
    li_num = 1 if logical_inversion_bool else 0
    pr_num = 1 if person_replacement_bool else 0

    return_problems = []
    for p in problem_list:
        changed_list = [] # 同じものいれないように
        label = int(p[0])
        t1 = p[1]
        t2 = p[2]
        for li in range(li_num+1):
            for pr in range(pr_num+1):
                label = int(p[0])
                tmp = t2
                if li and label == 1:
                    inversion_t2 = data_augumentation.make_logical_inversion(tmp)
                    if inversion_t2 != tmp:
                        tmp = inversion_t2
                        label = 0

                if pr:
                    replaced_t2 = data_augumentation.person_replacement(tmp)
                    if replaced_t2 != tmp and replaced_t2 != 'none':
                        tmp = replaced_t2

                if li and pr:
                    print(tmp)

                if tmp not in changed_list:
                    return_problems.append([label, t1, tmp])
                    changed_list.append(tmp)

    return return_problems

if __name__=='__main__':
    years = ['H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29','H30','R01','R02','R03']
    test_year = ['H18','R03']
    problems = main(['R03'])

    file = open('load_problems_test.csv','w')
    csv.writer(file).writerows(problems)
    file.close()
