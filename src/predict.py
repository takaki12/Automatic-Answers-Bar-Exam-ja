# ファインチューニングしたモデルを試す用

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

model_dir = '' # 試したいモデルのパス
model_name = model_dir + '/model_transformers'
tokenizer = AutoTokenizer.from_pretrained('') # トークナイザ―はpretrainedに合わせる
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# N
text1 = '借主は、貸主の承諾をえなければ、第三者に借用物の使用又は収益をさせることができない。'
text2 = 'Ａは、Ｂとの間で、Ａが所有する甲建物をＢに使用させる旨の使用貸借契約を締結した。Ｂは、Ａの承諾がなくても、甲建物を第三者に使用させることができる。'

input_text = text1 + tokenizer.sep_token + text2
print('t1',tokenizer.tokenize(text1))
print('t2',tokenizer.tokenize(text2))
print(len(tokenizer.tokenize(text1 + text2))+1)
encoding = tokenizer.encode_plus(
    input_text,
    return_tensors="pt"
)

with torch.no_grad():
    output = model(encoding['input_ids']).logits
    print(output)
    zero = output[0][0]
    one = output[0][1]
    if zero > one: 
        print("N")
    else:
        print("Y")
