# Automatic-Answers-Bar-Exam-ja
日本の司法試験(民法短答式)の自動解答をするシステムです。  

自動解答は、以下の問題サンプルのような入力に対し、labelを予測するタスクです。

>（pair id="H18-2-2" label="Y"）  
>（article）  
> 第六百九十八条　管理者は、本人の身体、名誉又は財産に対する急迫の危害を免れさせるために事務管理をしたときは、悪意又は重大な過失があるのでなければ、これによって生じた損害を賠償する責任を負わない。  
>（problem）  
> 車にひかれそうになった人を突き飛ばして助けたが，その人の高価な着物が汚損した場合，着物について損害賠償をする必要はない。  

## フォルダ構造
<pre>
.
├── data : システムに使うデータを保管する.
├── src : ソースコード
├── pretrained_model_note.txt : 事前学習済みモデルのメモ
└── README.md
</pre>

## ソースコード
<pre>
src
├── preprocessing : データの前処理用
    ├── calculate_similarity.py 文章の類似度を計算する
    ├── data_augmentation.py : データ拡張
    ├── extract_svo.py : 主語、述語、目的語を抽出する
    ├── kansuji_converter.py : 漢数字を英数字に変換
    ├── legal_datalist.py : 前処理で必要になるリストや辞書の管理
    ├── load_problems.py : データを読み込む.
    └── preprocess.py : データの前処理をする
├── data_module.py : データローダーを作成
├── main.py : データの読み込みとfine-tuningをする
├── model.py : モデル定義
├── predict.py : fine-tuningしたモデルの単問予測
└── train.py : 訓練部分のベースコード
</pre>

preprocessingフォルダでは、ファインチューニングするためのデータ処理を行っています。  
・データ例
```
<pair id="R04-01-I" label="N">
<t1>
（未成年者の法律行為）
第五条　未成年者が法律行為をするには、その法定代理人の同意を得なければならない。ただし、単に権利を得、又は義務を免れる法律行為については、この限りでない。
２　前項の規定に反する法律行為は、取り消すことができる。
（未成年者の営業の許可）
第六条　一種又は数種の営業を許された未成年者は、その営業に関しては、成年者と同一の行為能力を有する。
</t1>
<t2>
営業を許された未成年者がした法律行為は、その営業に関しないものであっても、取り消すことができない。
</t2>
</pair>
```  

load_problems.pyでは、上の例のようなファイルを読み込み、問題番号をキーとして、ラベル、関連条文(<t1>にあたる)と問題文(<t2>)を値とする辞書を作成します。　　
data_augmentation.pyでは、文末表現をあらかじめ設定しておいたペアと書き換えてラベルを反転させる処理(否定形生成)や一部の人物語を置き換える処理(人物語置き換え)をする関数です。  
```
否定形生成
「その営業に関しては、成年者と同一の行為能力を有 する。」 → 「その営業に関しては、成年者と同一の行為能力を有 しない。」
人物語置き換え
「営業を許された未成年者がした法律行為は、その営業に関しないものであっても、取り消すことができない。」　　
未成年者を置き換える
「営業を許されたAがした法律行為は、その営業に関しないものであっても、取り消すことができない。」
```  

select_article.pyでは、条文(<t1>)部分がとても長くなっている場合があります。(512トークンを超えることも)  
この問題に対処するため、複数の条文が指定されていた場合は、条や項ごとに分割して、Sentence-BERT/Sentece-LUKEを用いて文ベクトルを作成し、問題文と分割された条文の類似度を計算し、問題を解くために最も重要となる部分を選択します。  
分割するときに、下の例の2項目のように、"前項"という前の文章を必要とするものがあった場合に備えて、  

```
第五条　未成年者が法律行為をするには、その法定代理人の同意を得なければならない。ただし、単に権利を得、又は義務を免れる法律行為については、この限りでない。
２　前項の規定に反する法律行為は、取り消すことができる。
```  

各項単体で比較するのではなく、2つの項を組み合わせた文章とも比較しています。  
また、問題文と条文の文末表現が反対で、それが文章間の距離を遠ざけてしまわないように、文末表現を述語抽出をする処理(extract_svo.py)によって削除しました。

fine_tuning.pyでBERTやLUKEのような事前学習済みモデルを、用意したデータでファインチューニングし、テストデータに対する結果の予測と正答率判定ができます。
