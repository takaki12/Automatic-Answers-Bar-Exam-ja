# 訓練・検証・テストに使うデータを作成する
# 1. データ加工と読み込み
# 2. 条文選択

# GPU指定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import csv
import pickle
import itertools
from transformers import AutoTokenizer
import load_problems
import extract_svo
import calculate_similarity

# tokenizerを設定 *学習するモデルを揃える!!
tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-base")
output_ver = 'luke' # モデルの種類で設定する

# データを読み込み、リストを作成
years = ['H18','H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29', 'H30','R01','R02', 'R03', 'R04']
for year in years:
    print(year)

    # [label, article_text, problem_text, number, article_dict]のリスト
    problems = load_problems.load_problems(year)

    # 一時保存
    if not os.path.exists("data/interim/problems_" + output_ver + "_pkl"):
        os.mkdir("data/interim/problems_" + output_ver + "_pkl")

    with open("data/interim/problems_" + output_ver + "_pkl/riteval_" + year + "_jp.pkl", 'wb') as fp:
        pickle.dump(problems, fp)

    # 条文が長い場合(512超)のために短くする
    # 複数の条文があった場合、全ての組合せを作成して、問題文を解くために最も重要な組合せを洗濯する.
    coliee_pair = []
    for problem in problems:
        label = int(problem[0])
        article_text = problem[1]
        problem_text = problem[2]
        number = problem[3]
        article_dict = problem[4]

        tokens = len(tokenizer.tokenize(problem_text))
        limited_token = 512 - 3 - tokens # 512 - sp_tokens - probelem_token

        # 問題ごとの全ての条文をリストで格納
        article_list = []
        for num, articles in article_dict.items():
            for article in articles:
                article = article.replace('\n', '') # 改行が残っていた場合は消す
                article_list.append(article)

        sentences = []
        # bit全探索で全ての組合せを作成
        ptn = sorted(list(itertools.product([0,1], repeat=len(article_list))), reverse=True)
        for i, bits in enumerate(ptn):
            sentence = ""

            if list(bits).count(0) == len(article_list): # どの条文も使わないパターンは除外
                continue

            if i > 2**15: # 多くなりすぎないように制御
                break

            for i, b in enumerate(list(bits)):
                if b == 1:
                    sentence += article_list[i]

            if len(tokenizer.tokenize(sentence)) <= limited_token: # トークン数が512を超えていない場合
                sentences.append(sentence)

        # 文末表現により文類似度が変わらないように最後の述語は消去する
        query_text = extract_svo.delete_predicate(problem_text)
        queries = [query_text]
        
        # cos類似度により、sentencesの中で最も類似度の高い文を選択. (Sentence-LUKEを使う)
        article_top = calculate_similarity.compare_sentence(sentences, queries, model_type='luke', closest_n=1)

        # 最終的なpairを確定
        coliee_pair.append([label, article_top[0][0], article_top[0][1], problem_text, number])
        
    # 出力
    if not os.path.exists("data/processed/processed_data_" + output_ver):
        os.mkdir("data/processed/processed_data_" + output_ver)

    with open("data/processed/processed_data_" + output_ver + "/riteval_" + year + "_jp.csv", 'w') as f:
        csv.writer(f).writerows(coliee_pair)
