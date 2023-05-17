# textpairclassificationをする
# cross_validationで評価するコード

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import csv
import pickle
import pandas as pd
from sklearn.model_selection import KFold

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from transformers import AutoTokenizer

import data_augumentation

# データセットの作成
class DatasetGenerator(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row['text']
        labels = data_row['label']

        encoding = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels)
        )

# データローダの作成
class DataModuleGenerator(pl.LightningDataModule):
    """
    DataFrameからDataModuleを作成
    """
    def __init__(self, train_df, val_df, test_df, tokenizer, max_length=512, batch_size={'train':32, 'val':256, 'test':256}):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_batch_size = batch_size['train']
        self.val_batch_size = batch_size['val']
        self.test_batch_size = batch_size['test']
        
    def setup(self, stage=None):
        self.train_dataset = DatasetGenerator(self.train_df, self.tokenizer, self.max_length)
        self.val_dataset = DatasetGenerator(self.val_df, self.tokenizer, self.max_length)
        self.test_dataset = DatasetGenerator(self.test_df, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)
    
class ModelForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, result_output = './scores.csv', lr=1e-5):
        """_summary_

        Args:
            model_name (str): 事前学習済みbertモデル
            num_labels (int): 分類ラベル数
            lr (float): 学習率
        """

        super().__init__()
        self.save_hyperparameters()
        self.model_sc = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels
        )
        self.result_output = result_output

    def training_step(self, batch, batch_idx):
        output = self.model_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.model_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        scores = output.logits.tolist()

        # 結果の出力
        with open(self.result_output + '/scores.csv', 'a') as f:
            csv.writer(f).writerows(scores)

        # Accuracy
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0)
        self.log('accuracy', accuracy)
    
    # オプティマイザ
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def apply_data_augumentation(datalist):
    """データ拡張する関数

    Args:
        datalist (list): [label, t1, t2]
    """
    return_datalist = []
    for data in datalist:
        label = int(data[0])
        t1_text = data[1]
        t2_text = data[2]
        
        tmp_list = []
        for inversion_bool in [True, False]:
            for replace_person_bool in [True, False]:
                tmp = t2_text
                label = int(data[0])
                if inversion_bool and label == 1:
                    inversion_t2 = data_augumentation.make_logical_inversion(tmp)
                    if tmp != inversion_t2:
                        tmp = inversion_t2
                        label = 0

                if replace_person_bool:
                    replaced_t2 = data_augumentation.person_replacement(tmp)
                    if replaced_t2 != 'none' and tmp != replaced_t2:
                        tmp = replaced_t2

                if tmp not in tmp_list:
                    tmp_list.append(tmp)
                    return_datalist.append([label, t1_text, tmp])

    return return_datalist

def make_dataset(datalist, tokenizer):
    dataset = {'text':[], 'label':[]}
    for d in datalist:
        text = d[1] + tokenizer.sep_token + d[2]
        label = int(d[0])
        dataset['text'].append(text)
        dataset['label'].append(label)
    return dataset

def finetuning(output_dir, model_name, tokenizer, train_df, val_df, test_df):
    # Checkpoint
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=output_dir + '/model/'
    )

    # Early_stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        verbose=True, 
        mode="min", 
        patience=3
    )

    # Logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=output_dir + "/logs/"
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=10,
        callbacks = [checkpoint, early_stopping],
        logger=tb_logger
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
    accuracy = test[0]["accuracy"]
    print(f'Accuracy: {test[0]["accuracy"]:.2f}')

    # モデルの保存
    """finetuned_model = ModelForSequenceClassification_pl.load_from_checkpoint(
        checkpoint.best_model_path
    )

    finetuned_model.model_sc.save_pretrained(output_dir + '/model_transformers')"""

    return accuracy

if __name__=='__main__':
    # Setting ------------
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    #model_name = 'ku-nlp/deberta-v2-base-japanese'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_years = ['R02']
    #data_dir = '/home/tonaga/PycharmProjects/COLIEE2023/data/processed/coliee_data_top1'
    data_dir = '/home/tonaga/PycharmProjects/COLIEE2023/data/processed/coliee_data_top1'
    output_dir = '/home/tonaga/PycharmProjects/COLIEE2023/bert_05/bert_R02_wwm'
    # --------------------
    os.makedirs(output_dir, exist_ok=False)
    setting_log = ""
    setting_log += 'Model_name: ' + model_name + '\n'
    setting_log += 'tokenizer : ' + model_name + '\n'
    setting_log += 'test_year : ' + ','.join(test_years) + '\n'
    with open(output_dir + '/setting_note.txt','w') as f:
        f.writelines(setting_log)

    years = ['H18','H19','H20','H21','H22','H23','H24','H25','H26','H27','H28','H29', 'H30','R01','R02','R03','R04']
    available_years = [y for y in years if y not in test_years]
    kf_num = int(len(available_years)/2)

    # テストデータを作成
    test_datalist = []
    for year in test_years:
        #with open(data_dir + '/riteval_' + year + '_jp.csv', 'r') as f:
        with open(data_dir + "/riteval_" + year + '_jp.csv', 'r') as f:
            for row in csv.reader(f):
                test_datalist.append([row[1], row[2], row[4], row[0]])
    test_df = pd.DataFrame(make_dataset(test_datalist, tokenizer))

    kf = KFold(n_splits=kf_num, shuffle=False)
    cnt = 0
    for train, val in kf.split(available_years):
        output_dir2 = output_dir + '/' + str(cnt)
        os.makedirs(output_dir2)

        train_years = [available_years[i] for i in train]
        val_years = [available_years[i] for i in val]
        print("Train:", train_years)
        print("Val  :", val_years)

        finetuning_log = 'Model:' + str(cnt) + '\n'
        finetuning_log += 'train_years: ' + ','.join(train_years) + '\n'
        finetuning_log += 'val_years  : ' + ','.join(val_years) + '\n'
        finetuning_log += 'test_years : ' + ','.join(test_years) + '\n\n'

        print("Train:", train_years)
        print("Val  :", val_years)

        finetuning_log = 'Model:' + str(cnt) + '\n'
        #finetuning_log += 'train_years: ' + ','.join(train_years) + '\n'
        #finetuning_log += 'val_years  : ' + ','.join(val_years) + '\n'
        finetuning_log += 'test_years : ' + ','.join(test_years) + '\n\n'

        train_datalist = []
        # 民法条文より訓練データ作成
        with open("/home/tonaga/PycharmProjects/COLIEE2023/data/processed/minpou_dataset_tadashi.csv", 'r') as f:
            for row in csv.reader(f):
                train_datalist.append([row[0], row[1], row[2]])

        # 問題文より訓練データ作成
        for year in train_years:
            with open(data_dir + '/riteval_' + year + '_jp.csv', 'r') as f:
                year_datalist = []
                for row in csv.reader(f):
                    year_datalist.append([row[1], row[2], row[4]])

        # データ拡張
        datalist_pro = apply_data_augumentation(year_datalist)
        for row in datalist_pro:
            train_datalist.append(row)
        
        train_df = pd.DataFrame(make_dataset(train_datalist, tokenizer))

        # 検証データの作成
        val_datalist = []
        for year in val_years:
            with open(data_dir + '/riteval_' + year + '_jp.csv', 'r') as f:
                year_datalist = []
                for row in csv.reader(f):
                    year_datalist.append([row[1], row[2], row[4]])

        # データ拡張
        datalist_pro = apply_data_augumentation(year_datalist)
        for row in datalist_pro:
            val_datalist.append(row)
        val_df = pd.DataFrame(make_dataset(val_datalist, tokenizer))

        with open(output_dir2 + '/train.csv','w') as f:
            csv.writer(f).writerows(train_datalist)
        
        with open(output_dir2 + '/val.csv','w') as f:
            csv.writer(f).writerows(val_datalist)

        with open(output_dir2 + '/test.csv', 'w') as f:
            csv.writer(f).writerows(test_datalist)

        finetuning_log += 'data\n'
        finetuning_log += 'train_num: ' + str(len(train_datalist)) + '\n'
        finetuning_log += 'val_num  : ' + str(len(val_datalist)) + '\n'
        finetuning_log += 'test_num : ' + str(len(test_datalist)) + '\n\n'
        
        accuracy = finetuning(output_dir2, model_name, tokenizer, train_df, val_df, test_df)
        finetuning_log += 'Accuracy: ' + str(accuracy)
        with open(output_dir2 + '/finetuning_note.txt','w') as f:
            f.writelines(finetuning_log)
        cnt += 1

        # 結果の出力
        resuls = []
        with open(output_dir2 + "/scores.csv", 'r') as f:
            for i, row in enumerate(csv.reader(f)):
                pred_label = 0
                if float(row[1]) > float(row[0]):
                    pred_label = 1

                if pred_label == int(test_datalist[i][0]):
                    resuls.append([test_datalist[i][3], int(test_datalist[i][0]), pred_label, 'True', test_datalist[i][1], test_datalist[i][2]])
                else:
                    resuls.append([test_datalist[i][3], int(test_datalist[i][0]), pred_label, 'False', test_datalist[i][1], test_datalist[i][2]])

        with open(output_dir2 + "/results.csv", 'w') as f:
            csv.writer(f).writerows(resuls)

    
    result_dict = {}
    num_list = []
    label_dict = {}
    prob_dict = {}
    for i in range(kf_num):
        file = open(output_dir + '/' + str(i) + '/results.csv','r')
        for row in csv.reader(file):
            if row[0] not in result_dict.keys():
                num_list.append(row[0])
                label_dict[row[0]] = int(row[1])
                result_dict[row[0]] = [0, 0]
                prob_dict[row[0]] = [row[4], row[5]]
            result_dict[row[0]][int(row[2])] += 1

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
    print("ensemble_acc:", tcnt / (tcnt + fcnt))

    with open(output_dir + '/ensemble_result.csv', 'w') as f:
        csv.writer(f).writerows(ensemble_list)

    print(output_dir)
