# 2値分類finetuningのベースコード

# GPU指定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd

from transformers import BertJapaneseTokenizer, AutoTokenizer 
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from model import ModelForSequenceClassification_pl
from data_module import DataModuleGenerator
from preprocessing.load_problems import load_problems

# Setting ------------
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_year = ['H18','H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29', 'H30','R01','R02']
val_year = ['R03']
test_year = ['R04']
output_dir = 'output/bert_testR04'
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

def main():
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
        devices = 1, # 1で個数[1]で番号も指定
        max_epochs = 10, # 最大エポック数
        callbacks = [checkpoint, early_stopping],
        logger = tb_logger
    )

    # データの読み込み
    train_datalist = load_problems(train_year)
    val_datalist = load_problems(val_year)
    test_datalist = load_problems(test_year)

    train_dataset = make_dataset(train_datalist, tokenizer)
    val_dataset = make_dataset(val_datalist, tokenizer)
    test_dataset = make_dataset(test_datalist, tokenizer)

    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    test_df = pd.DataFrame(test_dataset)

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

    print(f'Accuracy: {test[0]["accuracy"]:.2f}')

    # ファインチューニングしたモデルのロード
    finetuning_model = ModelForSequenceClassification_pl.load_from_checkpoint(
        checkpoint.best_model_path
    )

    # モデルの保存
    finetuning_model.model_sc.save_pretrained(output_dir + '/model_finetuning/')

if __name__=='__main__':
    main()