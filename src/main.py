# 2値分類finetuningのコード. (基本構造はtrain.pyと同じ)
# データ拡張をしている.
# (交差検証のイメージ)2年ごとに訓練・検証データを変えた複数個のモデルを作成する.

# GPU指定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import pandas as pd
from sklearn.model_selection import KFold # 交差検証用

from transformers import BertJapaneseTokenizer, AutoTokenizer 
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from model import ModelForSequenceClassification_pl
from data_module import DataModuleGenerator
from preprocessing.load_problems import load_problems
from preprocessing import data_augmentation

# Setting ------------
model_name = 'studio-ousia/luke-japanese-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
years = ['H18','H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29', 'H30','R01','R02','R03', 'R04']
test_years = ['R04']
data_dir = 'data/processed/processed_data_luke' # preprocess.pyの処理後のデータセットを使う
output_dir = 'output/luke_testR04'
# --------------------

# listにある2文をsepトークンでつないで辞書{'text':[], 'label':[]}を作成
def make_dataset(datalist, tokenizer):
    """
    Args:
        datalist (list): データのリスト. [ラベル, 文1, 文2]
        tokenizer (tokenizer): tokenizerを指定

    Returns:
        dict: {'text':[], 'label':[]}
    """
    dataset = {'text':[], 'label':[]}
    for d in datalist:
        text = d[1] + tokenizer.sep_token + d[2]
        label = int(d[0])
        dataset['text'].append(text)
        dataset['label'].append(label)
    return dataset

def apply_data_augumentation(datalist, do_inversion=1, do_replacement=1):
    """データ拡張する関数

    Args:
        datalist (list): [label, t1, t2]
        do_inversion (int): 否定形拡張を適用するか
        do_replacement (int) : 人物置き換えを適用するか
    """
    return_datalist = []
    for data in datalist:
        label = int(data[0])
        t1_text = data[1]
        t2_text = data[2]
        
        tmp_list = []
        for inversion_bool in range(do_inversion+1):
            for replace_person_bool in range(do_replacement+1):
                tmp = t2_text
                label = int(data[0])

                # 否定形拡張
                if inversion_bool==1 and label == 1: # 先生「ラベルがNoの場合、文末を反転させても必ずYesになるかわからない」
                    inversion_t2 = data_augmentation.make_logical_inversion(tmp)
                    if tmp != inversion_t2:
                        tmp = inversion_t2
                        label = 0

                # 人物語置き換え
                if replace_person_bool == 1:
                    replaced_t2 = data_augmentation.person_replacement(tmp)
                    if replaced_t2 != 'none' and tmp != replaced_t2: # 人物語がなかった場合は何もしない
                        tmp = replaced_t2

                if tmp not in tmp_list: # 同じペアが作られないようにする
                    tmp_list.append(tmp)
                    return_datalist.append([label, t1_text, tmp])

    return return_datalist

# ファインチューニングをする
def finetuning(output_dir, model_name, tokenizer, train_df, val_df, test_df, model_save=True):
    """ファインチューニングをして結果を返す

    Args:
        output_dir (str): 出力先
        model_name (str): 事前学習済みモデル名
        tokenizer (tokenizer): 事前学習済みモデルと関連するtokenizerインスタンス
        train_df (dataframe): 訓練データ
        val_df (dataframe): 検証データ
        test_df (dataframe)): テストデータ
        model_save (bool): ファインチューニング後のモデルを保存するかどうか. Defaults to True.
    """

    # Checkpoint. 検証データのlossの最小値を保存する.
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'min',
        save_top_k = 1,
        save_weights_only = True,
        dirpath = output_dir + '/model/'
    )

    # Early_stopping. 検証データのlossが5回下がらなければ早期終了する.
    early_stopping = EarlyStopping(
        monitor = 'val_loss', 
        verbose = True, 
        mode = "min", 
        patience = 5
    )

    # Loggerの出力先を変更する.
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=output_dir + "/logs/"
    )

    # trainerの設定
    trainer = pl.Trainer(
        accelerator = 'gpu', 
        devices = 1, # 1で個数[1]で番号も指定
        max_epochs = 10, # 最大エポック数
        callbacks = [checkpoint, early_stopping],
        logger = tb_logger
    )

    # DataFrameを作成し、setupする
    data_module = DataModuleGenerator(
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df, 
        tokenizer=tokenizer, 
        max_length=512,
        batch_size={'train':32, 'val':256, 'test':256}
    )
    data_module.setup()

    # モデル定義
    sc_model = ModelForSequenceClassification_pl(
        model_name, 
        num_labels = 2, 
        result_output = output_dir,
        lr = 1e-5
    )

    # 学習開始!
    trainer.fit(
        sc_model, 
        data_module
    )

    print('ベストモデルのファイル:', checkpoint.best_model_path)
    print('ベストモデルの検証データに対する損失:', checkpoint.best_model_score)

    # テスト
    test = trainer.test(
        dataloaders=data_module.test_dataloader(), 
        ckpt_path=checkpoint.best_model_path
    )
    accuracy = test[0]["accuracy"]
    print(f'Accuracy: {test[0]["accuracy"]:.2f}')

    if model_save:
        # ファインチューニングしたモデルのロード
        finetuning_model = ModelForSequenceClassification_pl.load_from_checkpoint(
            checkpoint.best_model_path
        )

        # モデルの保存
        finetuning_model.model_sc.save_pretrained(output_dir + '/model_finetuning/')

    return accuracy

def main():
    # 設定をログとして残しておく
    setting_log = ""
    setting_log += 'Model_name: ' + model_name + '\n'
    setting_log += 'tokenizer : ' + model_name + '\n'
    setting_log += 'test_year : ' + ','.join(test_years) + '\n'
    with open(output_dir + '/setting_note.txt','w') as f:
        f.writelines(setting_log)

    # フォルダの上書き防止 -> 出力フォルダ作成
    if os.path.exists(output_dir):
        print("The path exists.")
        exit()
    os.makedirs(output_dir, exist_ok=True)

    # 訓練・検証に使うデータとテストに使うデータに分割
    available_years = [y for y in years if y not in test_years]

    # テストデータを作成
    test_datalist = []
    for year in test_years:
        with open(data_dir + "/riteval_" + year + '_jp.csv', 'r') as f:
            for row in csv.reader(f):
                test_datalist.append([row[0], row[1], row[3], row[4]])
    test_df = pd.DataFrame(make_dataset(test_datalist, tokenizer))

    # 分割数を決める. 2個ずつ分けたいため、2で割る.3年度分を検証にはしたくないため、使えるデータが奇数のときは分割数を+1する.
    if len(available_years) % 2 == 0:
        kf_num = len(available_years)/2
    else:
        kf_num = len(available_years)//2 + 1
    
    kf = KFold(n_splits=kf_num, shuffle=False) # 順番に処理したいため、shuffle=False
    cnt = 0 # モデル番号
    for train_num, val_num in kf.split():
        output_dir2 = output_dir + '/' + str(cnt)
        os.makedirs(output_dir2)

        train_years = [available_years[i] for i in train_num]
        val_years = [available_years[i] for i in val_num]
        print("Train:", train_years)
        print("Val  :", val_years)

        # fine-tuningログ part1:基本情報
        finetuning_log = 'Model:' + str(cnt) + '\n'
        finetuning_log += 'train_years: ' + ','.join(train_years) + '\n'
        finetuning_log += 'val_years  : ' + ','.join(val_years) + '\n'
        finetuning_log += 'test_years : ' + ','.join(test_years) + '\n\n'

        train_datalist = []
        # 問題文より訓練データ作成
        for year in train_years:
            with open(data_dir + '/riteval_' + year + '_jp.csv', 'r') as f:
                all_datalist_train = []
                for row in csv.reader(f):
                    all_datalist_train.append([row[0], row[1], row[3]])

        # データ拡張
        datalist_pro = apply_data_augumentation(all_datalist_train)
        for row in datalist_pro:
            train_datalist.append(row)
        
        train_df = pd.DataFrame(make_dataset(train_datalist, tokenizer))

        # 検証データの作成
        val_datalist = []
        for year in val_years:
            with open(data_dir + '/riteval_' + year + '_jp.csv', 'r') as f:
                all_datalist_val = []
                for row in csv.reader(f):
                    all_datalist_val.append([row[0], row[1], row[3]])

        # データ拡張
        datalist_pro = apply_data_augumentation(all_datalist_val)
        for row in datalist_pro:
            val_datalist.append(row)
        val_df = pd.DataFrame(make_dataset(val_datalist, tokenizer))

        # それぞれのデータを一応出力しておく
        with open(output_dir2 + '/train.csv','w') as f:
            csv.writer(f).writerows(train_datalist)
        
        with open(output_dir2 + '/val.csv','w') as f:
            csv.writer(f).writerows(val_datalist)

        with open(output_dir2 + '/test.csv', 'w') as f:
            csv.writer(f).writerows(test_datalist)

        # fine-tuningログ part2:データ数
        finetuning_log += 'data\n'
        finetuning_log += 'train_num: ' + str(len(train_datalist)) + '\n'
        finetuning_log += 'val_num  : ' + str(len(val_datalist)) + '\n'
        finetuning_log += 'test_num : ' + str(len(test_datalist)) + '\n\n'

        # 作成したデータからファインチューニングを実行
        accuracy = finetuning(output_dir2, model_name, tokenizer, train_df, val_df, test_df)

        # fine-tuningログ part3:accuracy
        finetuning_log += 'Accuracy: ' + str(accuracy)
        # 出力
        with open(output_dir2 + '/finetuning_note.txt','w') as f:
            f.writelines(finetuning_log)

        # 予測結果を出力する
        resuls = []
        with open(output_dir2 + "/scores.csv", 'r') as f: # [確率, 確率]というように出力されている
            for i, row in enumerate(csv.reader(f)):
                pred_label = 0
                # 確率の高い方を予測値として使う 0 or 1
                if float(row[1]) > float(row[0]):
                    pred_label = 1

                # 予測が合っていたか判定
                if pred_label == int(test_datalist[i][0]):
                    resuls.append([test_datalist[i][3], int(test_datalist[i][0]), pred_label, 'True', test_datalist[i][1], test_datalist[i][2]])
                else:
                    resuls.append([test_datalist[i][3], int(test_datalist[i][0]), pred_label, 'False', test_datalist[i][1], test_datalist[i][2]])

        with open(output_dir2 + "/results.csv", 'w') as f:
            csv.writer(f).writerows(resuls)

        cnt += 1

    # 複数のモデルの予測をアンサンブル(vote)して、最終的な結果をだす
    result_dict = {} # 結果保存用辞書
    num_list = [] # 問題番号
    label_dict = {} # 正解ラベル辞書 {問題番号:ラベル}
    prob_dict = {} # 予測値辞書 {問題番号:予測値}

    # 各モデルのresultsを調べる
    for i in range(kf_num):
        file = open(output_dir + '/' + str(i) + '/results.csv','r')
        for row in csv.reader(file):
            if row[0] not in result_dict.keys():
                num_list.append(row[0])
                label_dict[row[0]] = int(row[1])
                result_dict[row[0]] = [0, 0]
                prob_dict[row[0]] = [row[4], row[5]]
            # 各モデルによる問題ごとの予測値を投票
            result_dict[row[0]][int(row[2])] += 1

    # 各問題で0と1どちらが多いかを調べる (同じ場合は0とする)
    ensemble_list = []
    tcnt = 0
    fcnt = 0
    for num in num_list:
        zero = result_dict[num][0]
        one = result_dict[num][1]
        if zero >= one:
            pred = 0
        else:
            pred = 1

        if pred == label_dict[num]:
            ensemble_list.append([num, label_dict[num], pred, 'True', prob_dict[num][0], prob_dict[num][1]])
            tcnt += 1
        else:
            ensemble_list.append([num, label_dict[num], pred, 'False', prob_dict[num][0], prob_dict[num][1]])
            fcnt += 1
    file.close()
    # 最終的なAccuracy
    print("ensemble_acc:", tcnt / (tcnt + fcnt))

    with open(output_dir + '/ensemble_result.csv', 'w') as f:
        csv.writer(f).writerows(ensemble_list)

    print(output_dir)

if __name__=='__main__':
    main()