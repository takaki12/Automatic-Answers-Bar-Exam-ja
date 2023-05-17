# Sentence-BERT か Sentence-LUKE を使って、問題文を解くために重要な条文を選択する

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import csv
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy
import scipy.spatial
import itertools

# Sentence-Modelを定義
class SentenceModelJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)

# 問題文から述語を抽出し、消す。
import extract_svo # 文章中のSVOを抽出する
def delete_predicate(t2):
    svo_dict, svo_arrow_text = extract_svo.return_svo_triple(t2)
    predicate = svo_dict['述語']
    t2_no_predicate = t2[:t2.rfind(predicate)]
    return t2_no_predicate

if __name__=='__main__':
    MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
    # Sentence-BERT : sonoisa/sentence-bert-base-ja-mean-tokens-v2
    model = SentenceModelJapanese(MODEL_NAME) 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    drive_dir = './data/raw/COLIEE2023statute_data-Japanese/coliee2023_data_pkl'
    for file_name in os.listdir(drive_dir):
        if 'pkl' not in file_name:
            continue

        with open(drive_dir + '/' + file_name, 'rb') as fp:
            problem_list = pickle.load(fp)

        selected_list = []
        for problem in problem_list:
            number = problem[0]
            label = int(problem[1])
            article_dict = problem[2]
            article_dict_pro = problem[3]
            article_list = []
            all_articles = []
            for k,v in article_dict.items():
                for a in v:
                    article_list.append(a)
                    all_articles.append(a)

            article_list_pro = []
            for k,v in article_dict_pro.items():
                for a in v:
                    article_list_pro.append(a)
                    all_articles.append(a)

            t2_text = problem[4]
            t2_tokens = len(tokenizer.tokenize(t2_text))

            sentences = [] # 条文の組合せリスト
            i = 0
            for article_text in all_articles:
                if len(tokenizer.tokenize(article_text)) <= 509 - t2_tokens:
                    if article_text not in sentences: # かぶりは捨てる
                        sentences.append(article_text)

            for bits in itertools.product([0,1], repeat=len(article_list)):
                sentence = ''
                if len(sentences) >= 2**15:
                    break

                if list(bits).count(0) == len(article_list): # 空っぽは捨てる
                    continue

                for b, article in zip(bits, article_list):
                    if b == 1:
                        sentence += article

                # 512トークン超えないように...
                if len(tokenizer.tokenize(sentence)) <= 509 - t2_tokens:
                    if sentence not in sentences: # かぶりは捨てる
                        sentences.append(sentence)

            sentence_vectors = model.encode(sentences)
            # print(tokenizer.tokenize(sentences[0]))

            # 述語消す
            #t2_del_predicate = delete_predicate(t2_text)
            queries = [delete_predicate(t2_text)]
            query_embeddings = model.encode(queries).numpy()

            # 文ベクトルを基に問題文と条文の類似度を求める
            closest_n = 1
            for query, query_embedding in zip(queries, query_embeddings):
                distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric='cosine')[0]

                results = zip(range(len(distances)), distances)
                results = sorted(results, key=lambda x: x[1])

                print("\n\n======================\n\n")
                print("Query:", query)
                print("\nTop " + str(closest_n) +" most similar sentences in corpus:")

                for idx, distance in results[:closest_n]:
                    score = distance / 2
                    print(sentences[idx].strip(), "(Score: %.4f)" % score)
                    selected_list.append([number, label, sentences[idx].strip(), score, t2_text])

        # 選んだものをcsv形式で出力する
        with open("./data/processed/coliee_data_top1_nocomb/" + file_name.replace('.pkl', '.csv'), 'w') as f:
            csv.writer(f).writerows(selected_list)
