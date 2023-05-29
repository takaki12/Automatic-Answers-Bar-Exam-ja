# データ管理モジュール

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer

# データセットの作成
class DatasetGenerator(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """indexで指定したキーのデータを取得できるようにする.

        Args:
            index (_type_): _description_

        Returns:
            dict: エンコードされたテキストの辞書 {'input_ids':[], 'attention_mask':[], 'labels':tensor([])}
        """
        data_row = self.data.iloc[index]
        text = data_row['text']
        labels = data_row['label']

        # エンコードする
        encoding = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length', # 最大長分ない場合、[PAD]トークンを使う
            truncation=True, # 最大長を超えた場合、超えた分は除去
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(labels)
        )

# データローダの作成
class DataModuleGenerator(pl.LightningDataModule):
    """
    DataFrameからDataModuleを作成.
    DataFrameは、{'texxt':[str], 'label':[int]}でつくる.
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
        
    # train, val, test用のDatasetを準備. data_module作成後、必ず実行する必要がある.
    def setup(self, stage=None):
        self.train_dataset = DatasetGenerator(self.train_df, self.tokenizer, self.max_length)
        self.val_dataset = DatasetGenerator(self.val_df, self.tokenizer, self.max_length)
        self.test_dataset = DatasetGenerator(self.test_df, self.tokenizer, self.max_length)

    # 訓練用データローダー
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    # 検証用データローダー
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True)

    # テスト用データローダー
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)
