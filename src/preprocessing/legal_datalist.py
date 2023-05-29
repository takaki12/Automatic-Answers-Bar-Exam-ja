# よく使われるリストや辞書を管理する

import re
import csv

kansuji_list=['零','一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四','十五','十六','十七','十八','十九','二十','二十一','二十二','二十三','二十四','二十五']#漢数字のリスト

kansuji_dict = {'零':0, '一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9 }#漢数字と数字のペアリスト

LargeNumList = ['１','２','３','４','５','６','７','８','９'] # 大文字の数字のリスト
LargeNumDict = {1:'１',2:'２',3:'３',4:'４',5:'５',6:'６',7:'７',8:'８',9:'９'} # 半角数字 <-> 全角数字の辞書
LargeNumDict_re = {'１':1,'２':2,'３':3,'４':4,'５':5,'６':6,'７':7,'８':8,'９':9} # 全角数字 <-> 半角数字の辞書

Alphabet = ["Ａ","Ｂ","Ｃ","Ｄ","Ｅ","Ｆ","Ｇ","Ｈ","Ｉ","Ｊ","Ｋ","Ｌ","Ｍ","Ｎ","Ｏ","Ｐ","Ｑ","Ｒ","Ｓ","Ｔ","Ｕ","Ｖ","Ｗ","Ｘ","Ｙ","Ｚ"] # 全角アルファベット


# データ整形で使うデータリスト ----------
# 「次に掲げる〇〇」の〇〇を格納するリスト
nextlist = []
with open("data/external/next_list.txt", 'r') as f:
    for line in f.readlines():
        nextlist.append(re.sub("\n","",line))

# 「前項の規定」など、他の条や項を参照するとき用
okikae_nextlist=["の規定","の審判","と同様","の期間内"]
# 追加分
for x in nextlist:
    okikae_nextlist.append("の"+x)
    okikae_nextlist.append("の定める"+x)
    okikae_nextlist.append("に掲げる"+x)
    okikae_nextlist.append("に定める"+x)
    okikae_nextlist.append("において"+x)
    okikae_nextlist.append("に規定する"+x)
okikae_nextlist.sort(reverse=True, key=len)


# データ拡張で使うデータリスト ----------
# 否定語ペアのリスト
notlist = []
with open("data/external/inversion_verb.csv", 'r') as f:
    for row in csv.reader(f):
        notlist.append([row[0], row[1]])

# 人物語リスト
personlist = []
with open("data/external/persons.txt", 'r') as f:
    for line in f.readlines():
        personlist.append(line.replace('\n',''))
