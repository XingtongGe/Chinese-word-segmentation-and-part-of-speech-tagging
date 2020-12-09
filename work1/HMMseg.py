# 基于隐马尔可夫模型的统计分词方法

state_list = ['B', 'M', 'E', 'S']


def Viterbi(text, tran_1, initial_state, dict1, total_state):
    len_of_since = len(text)  # 观测序列长度
    max_p = [[0 for col in range(len(state_list))]
             for row in range(len_of_since)]
    path = [[0 for col in range(len(state_list))]
            for row in range(len_of_since)]
    # 初始状态
    if text[0] not in dict1:  # 字典中没有这个字
        max_p[0][0] = initial_state[0]*(1/total_state[0])
        max_p[0][3] = initial_state[1]*(1/total_state[3])
    else:
        max_p[0][0] = initial_state[0]*(dict1[text[0]][0]+1/total_state[0])
        max_p[0][3] = initial_state[1]*(dict1[text[0]][3]+1/total_state[3])
    # 迭代循环
    for i in range(1, len_of_since):
        max_item = [0.0, 0.0, 0.0, 0.0]
        for j in range(4):
            item = [0.0, 0.0, 0.0, 0.0]
            for k in range(4):
                if text[i] not in dict1:  # 字典中没有这个字
                    p = max_p[i-1][k]*(1/total_state[j])*tran_1[k][j]
                else:
                    p = max_p[i-1][k]*(dict1[text[i]][j] +
                                       1/total_state[j])*tran_1[k][j]
                item[k] = p
            max_item[j] = max(item)
            # 寻找item的最大值索引
            path[i][j] = item.index(max(item))
        max_p[i] = max_item
    newpath = []
    p = max_p[len_of_since-1].index(max(max_p[len_of_since-1]))
    newpath.append(p)
    for i in range(len_of_since-1, 0, -1):
        newpath.append(path[i][p])
        p = path[i][p]
    newpath.reverse()
    newtext = []
    j = 0
    for i in range(len_of_since):
        if newpath[i] == 3:
            newtext.append(text[i])
            j += 1
        elif newpath[i] == 2:
            newtext[j] += (text[i])
            j += 1
        elif newpath[i] == 1:
            newtext[j] += (text[i])
        elif newpath[i] == 0:
            newtext.append(text[i])

    return newtext


def train(trainset):
    # 状态转移概率矩阵
    tran = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    # 字典
    dict1 = {}
    for line in trainset:
        word = line.strip().split()
        len_of_sentence = len(word)
        num_of_char = len_of_sentence
        for character in word:
            num_of_char += (len(character)-1)
        # 当前句子中各字的状态序列
        list_of_state = []
        for character in word:
            if len(character) == 1:
                # 将该字加入字典
                if character not in dict1:
                    dict1[character] = [0.0, 0.0, 0.0, 1.0]
                else:
                    dict1[character][3] += 1
                # 加入该句子的状态序列
                list_of_state.append(3)
            elif len(character) > 1:
                len1 = len(character)
                # 词首的字加入字典
                if character[0] not in dict1:
                    dict1[character[0]] = [1.0, 0.0, 0.0, 0.0]
                elif character[0] in dict1:
                    dict1[character[0]][0] += 1
                # 词中的字加入字典
                for i in range(1, len1-1):
                    if character[i] not in dict1:
                        dict1[character[i]] = [0.0, 1.0, 0.0, 0.0]
                    elif character[i] in dict1:
                        dict1[character[i]][1] += 1
                # 词尾的字加入字典
                if character[len1-1] not in dict1:
                    dict1[character[len1-1]] = [0.0, 0.0, 1.0, 0.0]
                elif character[len1-1] in dict1:
                    dict1[character[len1-1]][2] += 1
                # 加入该句子的状态序列
                list_of_state.append(0)
                for i in range(len1-2):
                    list_of_state.append(1)
                list_of_state.append(2)
        # 修改状态转移概率矩阵
        for i in range(1, num_of_char):
            tran[list_of_state[i-1]][list_of_state[i]] += 1
    # print('未归一化的状态转移矩阵：')
    # print(tran)
    # 计算初始状态概率向量
    # 第一个字要么是B要么是S
    initial_state = [0.0, 0.0]
    total_state = [0.0, 0.0, 0.0, 0.0]
    for char in dict1:
        initial_state[0] += dict1[char][0]
        initial_state[1] += dict1[char][3]
        total_state[0] += dict1[char][0]
        total_state[1] += dict1[char][1]
        total_state[2] += dict1[char][2]
        total_state[3] += dict1[char][3]
    # 对状态转移矩阵和初始状态概率向量进行归一化
    tran_1 = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0], ]
    for i in range(4):
        sum = tran[i][0]+tran[i][1]+tran[i][2]+tran[i][3]
        for j in range(4):
            tran_1[i][j] = tran[i][j]/sum
    sum = initial_state[0]+initial_state[1]
    initial_state[0] = initial_state[0]/sum
    initial_state[1] = initial_state[1]/sum
    return tran_1, dict1, initial_state, total_state


if __name__ == '__main__':
    trainset = []
    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForSegementation/pku_training.utf8'
    with open(trainset_Path, 'r', encoding='utf-8') as f:
        for line in f:
            trainset.append(line)

    # 进行维特比算法
    text = '同胞们、朋友们、女士们、先生们：'
    tran_1, dict1, initial_state, total_state = train(trainset)
    newtext = Viterbi(text, tran_1, initial_state, dict1, total_state)
    print(newtext)
