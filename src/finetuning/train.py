# 2値分類finetuningのベースコード

# GPU指定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import csv
import pandas as pd

from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

import model

# Setting ------------
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_year = ['H18','H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29', 'H30','R01','R02']
val_year = ['R03']
test_year = ['R04']
output_dir = 'output/bert_testR04'
# --------------------

# フォルダの上書き防止 -> 出力フォルダ作成
if os.path.exists(output_dir):
    print("The path exists.")
    exit()
os.makedirs(output_dir, exist_ok=True)

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
    devices = 1,
    max_epochs=10,
    callbacks = [checkpoint, early_stopping],
    logger=tb_logger
)

# データの作成
train_datalist = []
# 民法読み込み
"""with open("/home/tonaga/PycharmProjects/COLIEE2023/data/processed/minpou_dataset_tadashi.csv", 'r') as f:
    for row in csv.reader(f):
        train_datalist.append(row)"""

# 問題読み込み
problem_dir = '/home/tonaga/PycharmProjects/COLIEE2023/data/processed/coliee_data_selected'
for y in train_year:
    file = open(problem_dir + '/riteval_' + y + '_jp_top5.csv', 'r')
    for row in csv.reader(file):
        number = row[0]
        label = int(row[1])
        t1 = row[2]
        t2 = row[4]
        tmp_list = []
        for x in [True, False]:
            for y in [True, False]:
                tmp = t2
                if x and label == 1: # 否定
                    inversion_t2 = data_augumentation.make_logical_inversion(tmp)
                    if inversion_t2 == tmp:
                        pass
                    else:
                        tmp = inversion_t2
                        label = 0

                if y: # 人物
                    replaced_t2 = data_augumentation.person_replacement(tmp)
                    if tmp != 'none' or replaced_t2 == tmp:
                        pass
                    else:
                        tmp = replaced_t2

                if tmp not in tmp_list:
                    tmp_list.append(tmp)
                    train_datalist.append([label, t1, tmp])
    file.close()

# 検証用
val_datalist = []
for y in val_year:
    file = open(problem_dir + '/riteval_' + y + '_jp_top5.csv', 'r')
    for row in csv.reader(file):
        number = row[0]
        label = int(row[1])
        t1 = row[2]
        t2 = row[4]
        tmp_list = []
        for x in [True, False]:
            for y in [True, False]:
                tmp = t2
                if x and label == 1: # 否定
                    inversion_t2 = data_augumentation.make_logical_inversion(tmp)
                    if inversion_t2 == tmp:
                        pass
                    else:
                        tmp = inversion_t2
                        label = 0

                if y: # 人物
                    replaced_t2 = data_augumentation.person_replacement(tmp)
                    if tmp != 'none' or replaced_t2 == tmp:
                        pass
                    else:
                        tmp = replaced_t2

                if tmp not in tmp_list:
                    tmp_list.append(tmp)
                    val_datalist.append([label, t1, tmp])    
    file.close()

# テスト
test_datalist = []
for y in test_year:
    file = open(problem_dir + '/riteval_' + y + '_jp_top5.csv', 'r')
    for row in csv.reader(file):
        number = row[0]
        label = int(row[1])
        t1 = row[2]
        t2 = row[4]
        test_datalist.append([label, t1, t2, number])
    file.close()

# それぞれのデータを出力
with open(output_dir + '/train.csv', 'w') as f:
    csv.writer(f).writerows(train_datalist)

with open(output_dir + '/val.csv', 'w') as f:
    csv.writer(f).writerows(val_datalist)

with open(output_dir + '/test.csv', 'w') as f:
    csv.writer(f).writerows(test_datalist)

def make_dataset(datalist: list):
    dataset = {'text':[], 'label':[]}
    for d in datalist:
        text = d[1] + tokenizer.sep_token + d[2]
        label = int(d[0])
        dataset['text'].append(text)
        dataset['label'].append(label)
    return dataset

train_dataset = make_dataset(train_datalist)
val_dataset = make_dataset(val_datalist)
test_dataset = make_dataset(test_datalist)

train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)
test_df = pd.DataFrame(test_dataset)

# DataFrameを作成し、setupする
data_module = model.DataModuleGenerator(
    train_df=train_df, 
    val_df=val_df, 
    test_df=test_df, 
    tokenizer=tokenizer, 
    max_length=512,
    batch_size={'train':32, 'val':256, 'test':256}
)
data_module.setup()

# モデル定義
sc_model = model.ModelForSequenceClassification_pl(
    model_name, 
    num_labels=2, 
    result_output=output_dir,
    lr=1e-5
)

# 学習開始
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
print(f'Accuracy: {test[0]["accuracy"]:.2f}')