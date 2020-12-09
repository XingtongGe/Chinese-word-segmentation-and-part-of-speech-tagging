import DictSeg
import random
import time
state_list = ['B', 'M', 'E', 'S']

start_time = 0


def prf_score(pre_data, gold_data):
    N_count = 0
    e_count = 0  # 将负类分为正
    c_count = 0  # 正类分为正
    e_line_count = 0
    c_line_count = 0

    for line1, line2 in zip(gold_data, pre_data):
        list1 = line1.split(' ')
        list2 = line2.split(' ')
        count1 = len(list1)  # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1  # 分对的行数
            c_count += count1  # 分对的词数
        else:
            e_line_count += 1
            arr1 = []
            arr2 = []
            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))  # list1中各个单词的起始位置
                pos += len(w)
            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))  # list2中各个单词的起始位置
                pos += len(w)
            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1
    end_time = time.time()
    R = float(c_count) / N_count
    P = float(c_count) / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count
    print("result:")
    print('标准词数：%d个，正确词数：%d个，错误词数：%d个' % (N_count, c_count, e_count))
    print('Recall: %f' % (R))
    print('Precision: %f' % (P))
    print('F MEASURE: %f' % (F))
    print('ERR RATE: %f' % (ER))
    print('共测试的行数为：%d行' % (e_line_count+c_line_count))
    print('分词测试耗费的时间为：%f秒' % (end_time-start_time))
    print('效率： %f行/秒' % ((c_line_count+e_line_count)/(end_time-start_time)))
    return F


if __name__ == '__main__':
    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForSegementation/pku_training.utf8'
    temp_trainset = []
    trainset = []
    testset = []
    goldset = []
    with open(trainset_Path, 'r', encoding='utf-8') as f:
        for line in f:
            temp_trainset.append(line)
    random.seed(1)
    random.shuffle(temp_trainset)
    pos = int(len(temp_trainset)*0.9)
    trainset = temp_trainset[:pos]  # 选取九成作为训练集
    goldset = temp_trainset[pos:]  # 选取一成作为测试集

    for line in goldset:
        word = line.strip().split()
        testset.append(''.join(word))

    dict1 = {}
    for line in trainset:
        if len(line) == 0:
            continue
        word = line.strip().split()
        for i in word:
            if i not in dict1:
                dict1[i] = 1

    res = []
    start_time = time.time()
    for line in testset:
        if len(line) == 0:
            continue
        res.append(' '.join(DictSeg.BIMM1(dict1, line)))
    prf_score(res, goldset)
