
goldset = []
testset = []
trainset_Path = 'D:/CODING/PythonCoding/nlp/dataset/test.utf8'

with open(trainset_Path, 'r', encoding='utf-8') as f:
    for line in f:
        goldset.append(line)

for line in goldset:
    word = line.strip().split()
    testset.append(''.join(word))

print(testset)
