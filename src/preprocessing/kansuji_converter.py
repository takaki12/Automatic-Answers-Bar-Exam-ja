number_dict = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '零':0}
disit_dict = {'十':10, '百':100, '千':1000, '万':10000, '億':100000000}

def convert_kansuji_to_number(kansuji):
    """
    引数の漢数字を数字(int型)に変換して返します。
    一から億(千億の位)まで対応!

    Args:
        kansuji (string): 漢数字
    """

    if kansuji == '':
        print("引数が空です!")
        exit(1)

    num = 0 # 1つの数字を保持する
    block_num = 0 # 1000までの数字( A千B百C十D )を保持する
    converted_number = 0 # 最終出力
    
    for k in kansuji:
        # 一から九であった場合
        if k in number_dict.keys():
            num = number_dict[k]

        # 桁文字(十、百、...)であった場合
        if k in disit_dict.keys():
            disit_num = disit_dict[k]

            # 桁が10000より上のとき
            if disit_num >= 10000:
                # イメージは、千の位までのブロック(A千B百C十D) * disit_num(万, 億)
                block_num = (block_num + num) * disit_num
                converted_number += block_num
                num = 0
                block_num = 0
            # 桁が1000より下のとき
            else:
                if num == 0:
                    num = 1 # 「千五」や「十」など、桁文字の前に数字がない場合の処理
                block_num += num * disit_num
                num = 0
    
    converted_number += block_num + num
    return converted_number

if __name__=='__main__':
    print("{:,}".format(convert_kansuji_to_number('七')))
    # 2
    print("{:,}".format(convert_kansuji_to_number('二十五')))
    # 25
    print("{:,}".format(convert_kansuji_to_number('百十一')))
    # 111
    print("{:,}".format(convert_kansuji_to_number('二千十三')))
    # 2,013
    print("{:,}".format(convert_kansuji_to_number('一万二千三百四十五')))
    # 12,345
    print("{:,}".format(convert_kansuji_to_number('九千九百九十九万九千九百九十九')))
    # 99,999,999
    print("{:,}".format(convert_kansuji_to_number('三億二千万一')))
    # 320,000,001
    print("{:,}".format(convert_kansuji_to_number('二十一億四千七百四十八万三千六百四十八')))
    # 2,147,483,647
    print("{:,}".format(convert_kansuji_to_number('九千九百九十九億九千九百九十九万九千九百九十九')))
    # 999,999,999,999
    print("{:,}".format(convert_kansuji_to_number('二億四千万の瞳')))