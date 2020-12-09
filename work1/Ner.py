# 命名实体识别

def train(trainset_Path):
    state_dict = {}
    states = []
    dict1 = {}
    with open(trainset_Path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) < 2:
                continue
            word = line.strip().split()
            if word[1] not in state_dict:
                state_dict[word[1]] = 1
            else:
                state_dict[word[1]] += 1
        for sta in state_dict:
            states.append(sta)
        print(len(states))
        print(states)
        f.seek(0)
        # 初始化初始状态向量
        initial_state = {}
        # 初始化状态转移概率矩阵
        tran = [[0.0 for i in range(len(states))] for j in range(len(states))]
        cur_state_list = []  # 存储一行的状态序列
        for line in f:
            if line == '\n':
                # 处理整行数据,顺便填充初始状态向量
                if cur_state_list[0] not in initial_state:
                    initial_state[cur_state_list[0]] = 1
                else:
                    initial_state[cur_state_list[0]] += 1
                for i in range(len(cur_state_list)-1):
                    tran[states.index(cur_state_list[i])][states.index(
                        cur_state_list[i+1])] += 1
                cur_state_list = []
                continue
            word = line.strip().split()
            if word[0] not in dict1:
                dict1[word[0]] = [0 for i in range(len(states))]  # 在词典中初始化该词
                dict1[word[0]][states.index(word[1])] += 1
            else:
                dict1[word[0]][states.index(word[1])] += 1
            cur_state_list.append(word[1])
        return dict1, state_dict, states, tran, initial_state


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
        res.append(states[newpath[i]])
    return res


if __name__ == '__main__':
    trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/DataSetForNER/trainNer.txt'
    dict1, state_dict, states, tran, initial_state = train(trainset_Path)
    # print(state_dict)
    # print(initial_state)
    text1 = '我是新闻中心主任，同时也是公司的执行董事。他是中国国籍的美国人，从中国人民大学毕业，大学本科学历。这位同学的学历是理学硕士，现在是一名注册会计师'
    text = []
    for i in range(len(text1)):
        text.append(text1[i])
    res = Viterbi(text, dict1, state_dict, states, tran, initial_state)
    print(res)
    cur_word_list = []
    end_words = []
    for i in range(len(res)):
        if res[i][0] == 'B':
            cur_word_list.append(text[i])
        elif res[i][0] == 'E':
            cur_word_list.append(text[i])
            end_words.append(cur_word_list)
            cur_word_list = []
        elif res[i][0] == 'M':
            cur_word_list.append(text[i])
        else:
            continue
    print('句中的命名实体有：')
    for line in end_words:
        print(''.join(line))
