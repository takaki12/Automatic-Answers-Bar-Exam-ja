# モデル定義モジュール

import csv
import torch
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification

# モデル定義. 2値分類のため、SequenceClassificationを使う.
class ModelForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, result_output = './output', lr=1e-5):
        """モデル読み込みと各ステップの処理を定義する

        Args:
            model_name (str): 事前学習済みbertモデル
            num_labels (int): 分類ラベル数
            result_output (str): 予測値の出力先
            lr (float): 学習率. 1e-5がdefault
        """

        super().__init__()
        self.save_hyperparameters()
        # モデルを読み込む. 違うモデルいも対応するため、BERTからAutoModelに変更.
        self.model_sc = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels
        )
        # 予測値の出力先
        self.result_output = result_output

    # train時の処理
    def training_step(self, batch, batch_idx):
        output = self.model_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    # val時の処理
    def validation_step(self, batch, batch_idx):
        output = self.model_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    # test時の処理
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.model_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        scores = output.logits.tolist()

        # 各予測値の出力
        with open(self.result_output + '/scores.csv', 'a') as f: # バッチごとの処理のため追記モード
            csv.writer(f).writerows(scores)

        # Accuracyの計算
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct/labels.size(0)
        self.log('accuracy', accuracy)
    
    # オプティマイザ 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
