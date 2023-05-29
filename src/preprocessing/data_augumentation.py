# データ拡張用のメソッドをまとめたモジュール

import os
import re
import csv
import LegalDataList

def make_logical_inversion(data):
    """
    否定形の文章を作成します。

    Args:
        data (str): 問題文
    """
    data = data.replace("\n","")
    data += '\n'
        
    kakko =  re.search("「.*」", data)
    if kakko:
        data = re.sub(kakko.group(), "KAKKO", data) # 括弧を一時削除

    tadashi = re.search("。ただし[^。]*。\n",data)
    if tadashi:
        data = re.sub(tadashi.group(),"。\n",data) # ただしを一時削除

    if not "。\n" in data:
        data = data.replace("\n", "。\n")
    
    processed_data = data

    changed = False
    for verb in LegalDataList.NotList:
        if (verb[0] + "。\n") in data:
            processed_data = data.replace((verb[0]+ "。\n"), verb[1]+ "。")
            changed = True
            break

        elif (verb[1] + "。\n") in data:
            processed_data = data.replace((verb[1] + "。\n"), verb[0]+ "。")
            changed = True
            break

    if not changed:
        print('error1022:', data)

    # 一時的に消していたものを戻す
    if tadashi:
        processed_data = re.sub("。\n", tadashi.group(), processed_data)
    
    if kakko:
        processed_data = re.sub("KAKKO", kakko.group(), processed_data)
    
    return processed_data

def person_replacement(data):
    """
    判例問題(AやBで表された人物が登場する問題)風の問題を作成.
    人物名を抽出すし、でてきた順にA,B,C...と置き換える.

    Args:
        data (str): 問題文
    """
    data = data.replace('\n','')
    processed_data = data

    for alphabet in LegalDataList.LARGEAB:
        if alphabet in data:
            # すでに判例問題ならなにもしない
            return "none"
    
    cnt = 0
    for person in LegalDataList.PersonList:
        if (person+"が") in processed_data or (person+"を") in processed_data or (person+"の") in processed_data or (person+"に") in processed_data or (person+"は") in processed_data or (person+"、") in processed_data or (person+"又") in processed_data:
            processed_data = processed_data.replace(person, LegalDataList.LARGEAB[-1] + str(cnt)) # 一時的にZ+n(1,2,3,...)に置換
            cnt += 1
    
    # 人物名がひとつも登場しない場合はここで終わり
    if cnt == 0:
        return "none"
    
    # 出てきた順に置き換え
    # 順番調べる
    z_dict = {}
    for i in range(cnt):
        key = LegalDataList.LARGEAB[-1] + str(i)
        z_dict[key] = processed_data.index(key)
    z_sorted = sorted(z_dict.items(), key=lambda x:x[1])
    #print(z_sorted)

    # 前から置き換え!
    for i, z in enumerate(z_sorted):
        target = z[0]
        processed_data = processed_data.replace(target, LegalDataList.LARGEAB[i])
    
    return processed_data


if __name__=='__main__':
    text = '売主が買主に代金の支払いを請求する。'
    print('original:', text)
    print('inversion:', make_logical_inversion(text))
    print('replace_person:', person_replacement(text))