import CaboCha
import sys
import codecs
# https://docs.python.org/ja/3.6/library/typing.html
from typing import Dict, Tuple, List

def return_svo_triple(sentence:str) -> Tuple[Dict, List]:
    c = CaboCha.Parser()
    tree = c.parse(sentence)
    size = tree.size()
    myid = 0
    ku_list = []
    ku = ''
    ku_id = 0
    ku_link = 0
    kakari_joshi = 0
    kaku_joshi = 0

    for i in range(0, size):
        token = tree.token(i)
        if token.chunk:
            if (ku!=''):
                ku_list.append((ku, ku_id, ku_link, kakari_joshi, kaku_joshi))  #前 の句をリストに追加

            kakari_joshi = 0
            kaku_joshi = 0
            ku = token.normalized_surface
            ku_id = myid
            ku_link = token.chunk.link
            myid=myid+1
        else:
            ku = ku + token.normalized_surface

        m = (token.feature).split(',')
        if (m[1] == u'係助詞'):
            kakari_joshi = 1
        if (m[1] == u'格助詞'):
            kaku_joshi = 1

    ku_list.append((ku, ku_id, ku_link, kakari_joshi, kaku_joshi))  # 最後にも前の句をリストに追加
    for k in ku_list:
        if (k[2]==-1):  # link==-1?      # 述語である
            jutsugo_id = ku_id  # この時のidを覚えておく
    #述語句
    predicate_word = [k[0] for k in ku_list if (k[1]==jutsugo_id)]
    #for k in ku_list:
    #   if (k[1]==jutsugo_id):  # jutsugo_idと同じidを持つ句を探す
    #       print(k[1], k[0], k[2], k[3], k[4])
    #述語句に係る句
    # jutsugo_idと同じidをリンク先に持つ句を探す
    word_to_predicate_list = [k[0] for k in ku_list if k[2]==jutsugo_id]
    #　述語句に係る句 -> 述語句
    svo_arrow_text = [str(word_to_predicate) + "->" + str(predicate_word[0]) for word_to_predicate in word_to_predicate_list]
    #print(svo_arrow_text)

    svo_dict = {}
    for num, k in enumerate(ku_list):
        if (k[2]==jutsugo_id):  # jutsugo_idと同じidをリンク先に持つ句を探す
            if (k[3] == 1):
                subject_word = k[0]
                svo_dict["主語"] = subject_word
                #print(subject_word)
            if (k[4] == 1):
                object_word = k[0]
                svo_dict["目的語"] = object_word
                #print(object_word)
        if (k[1] == jutsugo_id):
                predicate_word = k[0]
                svo_dict["述語"] = predicate_word
                #print(predicate_word)

    return (svo_dict, svo_arrow_text)

if __name__=='__main__':
    sentence = '成年被後見人Ａが未成年者Ｂの法定代理人としてした行為は、Ａの行為能力の制限によっては取り消すことができない。'
    svo_dict, svo_arrow_text = return_svo_triple(sentence)
    print(svo_dict)
    print(svo_arrow_text)
