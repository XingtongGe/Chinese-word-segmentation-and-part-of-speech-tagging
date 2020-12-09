# 对词性标注进行测试和评估
import HMMmark
import time

start_time = 0


def evalu(pre_data, gold_data):
    N_count = 0  # 将正类分为正或者将正类分为负
    e_count = 0  # 将负类分为正
    c_count = 0  # 正类分为正
    e_line_count = 0
    c_line_count = 0
    for list1, list2 in zip(gold_data, pre_data):
        if len(list1) != len(list2):
            print('长度不相同！')
        count1 = len(list1)  # 标准分词数
        N_count += count1
        if list1 == list2:
            c_line_count += 1  # 标注正确的行数
            c_count += count1  # 标注正确的词数
        else:
            e_line_count += 1
            for i in range(len(list1)):
                if list1[i] == list2[i]:
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
    print('测试的行数为：%d行' % (e_line_count+c_line_count))
    print('标注测试耗费的时间为：%f秒' % (end_time-start_time))
    print('效率： %f行/秒' % ((c_line_count+e_line_count)/(end_time-start_time)))

    return F


if __name__ == '__main__':
    testset = []
    goldset = []
    test_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForMark/199801-test.txt'
    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForMark/199801-train.txt'
    # 对测试集数据进行处理
    with open(test_Path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for i in range(len(words)):
                if '[' in words[i]:
                    words[i] = words[i][1:]
                elif ']' in words[i]:
                    newword = words[i].split(']')
                    words[i] = newword[0]
            goldset.append(words)

    for line in goldset:
        templist = []
        for word in line:
            newword = word.split('/')
            templist.append(newword[0])
        testset.append(templist)

    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForMark/199801-train.txt'
    dict1, state_list, states, tran, initial_state = HMMmark.train(
        trainset_Path)
    num_of_state = len(states)

    res = []
    start_time = time.time()

    for line in testset:
        res.append(HMMmark.Viterbi(
            line, dict1, state_list, states, tran, initial_state))

    evalu(res, goldset)
