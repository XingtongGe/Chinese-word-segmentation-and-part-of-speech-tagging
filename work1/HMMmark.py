# HMM 词性标注

# 状态字典和列表


# 状态数量

# 状态转移概率矩阵

# 初始状态概率向量

# 词典


def train(trainset_Path):
    state_list = {}
    states = []
    dict1 = {}
    with open(trainset_Path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()
            len_of_sentence = len(word)
            for k in range(0, len_of_sentence):
                words = word[k].split('/')
                if ']' in words[1]:
                    newwords = words[1].split(']')
                    for i in newwords:
                        if i not in state_list:
                            state_list[i] = 1
                        else:
                            state_list[i] += 1
                elif words[1] not in state_list:
                    state_list[words[1]] = 1
                else:
                    state_list[words[1]] += 1
                # print(words[1])
        for sta in state_list:
            states.append(sta)
        print(len(states))
        print(states)

        f.seek(0)
        # 初始化初始状态向量
        initial_state = {}
        # 初始化状态转移概率矩阵
        tran = [[0.0 for i in range(len(states))] for j in range(len(states))]
        # 构造词典
        for line in f:
            cur_state_list = []  # 记录当前句子的状态序列，用于填充状态转移矩阵
            word = line.strip().split()
            len_of_sentence = len(word)
            if len(word) == 0:
                continue
            # 填充初始状态向量
            cur_word = word[0].split('/')  # 句子中的第一个词
            if ']' not in cur_word[1]:
                if cur_word[1] in states:
                    if cur_word[1] not in initial_state:
                        initial_state[cur_word[1]] = 1
                    else:
                        initial_state[cur_word[1]] += 1
            else:
                new_curword = cur_word[1].split(']')
                if new_curword[0] in states:
                    if new_curword[0] not in initial_state:
                        initial_state[new_curword[0]] = 1
                    else:
                        initial_state[new_curword[0]] += 1

            for k in range(1, len(word)):
                words = word[k].split('/')
                if '[' not in words[0]:
                    if words[0] not in dict1:
                        dict1[words[0]] = [0 for i in range(
                            len(states))]  # 在词典中初始化该词
                        if ']' not in words[1]:
                            cur_state_list.append(words[1])
                            if words[1] in states:
                                dict1[words[0]][states.index(words[1])] = 1

                        else:
                            newwords = words[1].split(']')
                            cur_state_list.append(newwords[0])
                            if newwords[0] in states:
                                dict1[words[0]][states.index(newwords[0])] = 1
                    else:
                        if ']' not in words[1]:
                            cur_state_list.append(words[1])
                            if words[1] in states:
                                dict1[words[0]][states.index(words[1])] += 1
                        else:
                            newwords = words[1].split(']')
                            cur_state_list.append(newwords[0])
                            if newwords[0] in states:
                                dict1[words[0]][states.index(newwords[0])] += 1
                else:
                    newchar = words[0].split('[')
                    if newchar[1] not in dict1:
                        dict1[newchar[1]] = [0 for i in range(len(states))]
                        if ']' not in words[1]:
                            cur_state_list.append(words[1])
                            if words[1] in states:
                                dict1[newchar[1]][states.index(words[1])] = 1
                        else:
                            newwords = words[1].split(']')
                            cur_state_list.append(newwords[0])
                            if newwords[0] in states:
                                dict1[newchar[1]][states.index(
                                    newwords[0])] = 1
                    else:
                        if ']' not in words[1]:
                            cur_state_list.append(words[1])
                            if words[1] in states:
                                dict1[newchar[1]][states.index(words[1])] += 1
                        else:
                            newwords = words[1].split(']')
                            cur_state_list.append(newwords[0])
                            if newwords[0] in states:
                                dict1[newchar[1]][states.index(
                                    newwords[0])] += 1
            for i in range(len(cur_state_list)-1):
                if cur_state_list[i] in states and cur_state_list[i+1] in states:
                    tran[states.index(cur_state_list[i])][states.index(
                        cur_state_list[i+1])] += 1
    return dict1, state_list, states, tran, initial_state


def Viterbi(text, dict1, state_list, states, tran, initial_state):
    num_of_state = len(states)
    len_of_text = len(text)  # 观测序列长度
    max_p = [[0.0 for col in range(len(states))]for row in range(len_of_text)]
    path = [[0.0 for col in range(len(states))]for row in range(len_of_text)]
    # 初始状态
    if text[0] in dict1:
        for state in initial_state:
            max_p[0][states.index(state)] = initial_state[state]*(
                float(dict1[text[0]][states.index(state)]+1)/float(state_list[state]))
    else:
        for state in initial_state:
            max_p[0][states.index(state)] = initial_state[state] * \
                (float(1)/float(state_list[state]))

    # 迭代循环
    for i in range(1, len_of_text):
        max_item = [0 for temp in range(len(states))]
        for j in range(num_of_state):
            item = [0 for temp in range(len(states))]
            for k in range(num_of_state):
                if text[i] in dict1:
                    p = max_p[i-1][k]*(float(dict1[text[i]][j]+1) /
                                       float(state_list[states[j]]))*tran[k][j]
                else:
                    p = max_p[i-1][k] * \
                        (float(1)/float(state_list[states[j]]))*tran[k][j]
                item[k] = p
            max_item[j] = max(item)
            # 寻找item的最大值索引
            path[i][j] = item.index(max(item))
        max_p[i] = max_item
    newpath = []
    p = max_p[len_of_text-1].index(max(max_p[len_of_text-1]))
    newpath.append(p)
    for i in range(len_of_text-1, 0, -1):
        newpath.append(path[i][p])
        p = path[i][p]
    newpath.reverse()
    res = []
    for i in range(len_of_text):
        res.append(text[i]+'/'+states[newpath[i]])
    return res


if __name__ == '__main__':

    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForMark/199801-train.txt'
    dict1, state_list, states, tran, initial_state = train(trainset_Path)
    num_of_state = len(states)
    # 应用维特比算法
    text = ['我', '是', '他', '的', '朋友']
    res = Viterbi(text, dict1, state_list, states, tran, initial_state)
    print(res)
