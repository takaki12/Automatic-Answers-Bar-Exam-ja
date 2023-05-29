# Sentence-BERT / Sentence-LUKEを用いて文ベクトルを作成し、文の類似度を計算する.
# 学習済みモデルにはsonoisaさんの2つのモデルを使った.
# 参考: https://qiita.com/sonoisa/items/1df94d0a98cd4f209051
# BERT-base: sonoisa/sentence-bert-base-ja-mean-tokens-v2
# LUKE-base: sonoisa/sentence-luke-japanese-base-lite

import torch
from transformers import BertJapaneseTokenizer, BertModel
from transformers import AutoTokenizer, LukeModel
import scipy.spatial

class SentenceBertJapanese:
    def __init__(self, model_name="sonoisa/sentence-bert-base-ja-mean-tokens-v2", device=None):
        # モデルの読み込み
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

        # cudaの設定
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    # 文中の各単語ベクトルの平均をとる
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # model_outputの最初の要素に全てのトークン埋め込みが含まれる
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            # エンコードする
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input) # 文埋め込みの取得
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu') # poolingして文ベクトルに

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
    
# Sentence-BERTのLuke版がでていたので追加!
class SentenceLukeJapanese:
    def __init__(self, model_name="sonoisa/sentence-luke-japanese-base-lite", device=None):
        # モデルの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LukeModel.from_pretrained(model_name)
        self.model.eval()

        # cudaの設定
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    # 文中の各単語ベクトルの平均をとる
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # model_outputの最初の要素に全てのトークン埋め込みが含まれる
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            # エンコードする
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input) # 文埋め込みの取得
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu') # poolingして文ベクトルに

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
    
# 文章比較して類似度順にする
def compare_sentence(sentences, queries, model_type='bert', closest_n=5):
    """
    # sentenceとqueriesの文ベクトルをつくり、コサイン類似度を比較して、似ている文章順に並べる

    Args:
        sentences  (list) : 文リスト(分割した条文)
        queries    (list  : 問題文
        model_type (str)  : bertかlukeどちらを使いたいか
        closest_n  (int)  : 順番に並べた後で、上位何件ほしいか
    
    """
    if model_type == 'bert':
        model = SentenceBertJapanese()
    elif model_type == 'luke':
        model = SentenceLukeJapanese()
    else:
        print("No model type : " + model_type)
        exit()

    # sentencesとqueriesをベクトル化
    sentence_vectors = model.encode(sentences)
    query_embeddings = model.encode(queries).numpy()

    log = ''
    sentence_list = []
    for query, query_embedding in zip(queries, query_embeddings):
        # cos距離を求める
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric='cosine')[0]
        # cos類似度を求める
        cos_sim = 1.0 - distances

        # 結果をまとめる
        results = zip(range(len(cos_sim)), cos_sim)
        # cos類似度が高い順にソート
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # 上位closest_n件を取得
        for idx, score in results[:closest_n]:
            # print(sentences[idx].strip(), score)
            log += sentences[idx].strip() + " (Score: " +  str(score) + ")" + '\n'
            sentence_list.append([sentences[idx].strip(), score])

    return sentence_list

if __name__=='__main__':
    sentences = ["ある事業のために他人を使用する者は、被用者がその事業の執行について第三者に加えた損害を賠償する責任を負う。",\
                 "ただし、使用者が被用者の選任及びその事業の監督について相当の注意をしたとき、又は相当の注意をしても損害が生ずべきであったときは、この限りでない。",\
                 "使用者に代わって事業を監督する者も、前項の責任を負う。",\
                 "前二項の規定は、使用者又は監督者から被用者に対する求償権の行使を妨げない。"  ]
    queries = ["使用者は、被用者の選任及び監督について相当の注意をしたことを証明した場合、責任を免れる"]

    sentence_list = compare_sentence(sentences, queries)
    for sv, cs in sentence_list:
        print(sv, cs)