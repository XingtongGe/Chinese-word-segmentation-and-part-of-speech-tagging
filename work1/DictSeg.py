# 基于词典的分词
def FMM1(word_dict, text):
    window_size = max(map(len, [w for w in word_dict]))  # 找出词典中最长的一项的长度
    result = []
    index = 0
    text_size = len(text)
    while text_size > index:
        for size in range(window_size+index, index, -1):
            piece = text[index:size]
            if piece in word_dict:
                index = size - 1
                break
        index = index + 1
        result.append(piece)
    return result


def RMM1(word_dict, text):
    window_size = max(map(len, [w for w in word_dict]))  # 找出词典中最长的一项的长度
    result = []
    index = len(text)
    window_size = min(index, window_size)
    while index > 0:
        for size in range(index-window_size, index):
            piece = text[size:index]
            if piece in word_dict:
                index = size + 1
                break
        index = index - 1
        result.append(piece)
    result.reverse()
    return result


def BIMM1(word_dict, text):
    res_fmm = FMM1(word_dict, text)
    res_rmm = RMM1(word_dict, text)
    if len(res_fmm) == len(res_rmm):
        if res_fmm == res_rmm:
            return res_fmm
        else:
            f_word_count = len([w for w in res_fmm if len(w) == 1])
            r_word_count = len([w for w in res_rmm if len(w) == 1])
            return res_fmm if f_word_count < r_word_count else res_rmm
    else:
        return res_fmm if len(res_fmm) < len(res_rmm) else res_rmm


if __name__ == '__main__':
    dic_path = './nlp/dataset/DataSetForSegementation/pku_training.utf8'
    dict1 = {}
    with open(dic_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) == 0:
                continue
            word = line.strip().split()
            for i in word:
                if i not in dict1:
                    dict1[i] = 1
    text = "他是一个普通人"
    print(BIMM1(dict1, text))
