# -- coding: utf-8 --
import codecs
import math
import sys
import os

#input_file = "SogouC/Segment/C000020_pre.txt"
#ClassCode = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']

# classCodes = ['C000013','C000024']
#path after cutting
file_path = sys.path[0] + '/../Data/train'
# path = "trainData"
# Stop words
def isStopWord(word):
    with open ('stopwords.txt', 'r') as f:
        words = f.readlines()
        if word in words:
            return True
        else:
            return False

# 卡方计算公式
def Chi(a, b, c, d):
    return float(pow((a * d - b * c), 2)) /float((a + c) * (a + b) * (b + d) * (c + d))
# 对卡方检验所需的 a b c d 进行计算
# a：在这个分类下包含这个词的文档数量
# b：不在该分类下包含这个词的文档数量
# c：在这个分类下不包含这个词的文档数量
# d：不在该分类下，且不包含这个词的文档数量

#classDic... map the classcode with the class list 
#each class list item maps a doc
#each doc maps a set putting the words


def buildSets():
    fList = list()
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        for f in files:
            if f != ".DS_Store":
                fList.append(f) 
    classDocDic = dict()
    classWordDic = dict()
    for l in fList:
        path = file_path + '/' + l
        classDocList = list()
        classWordSet = set()
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                docWordSet = set()
                doc = line.strip('\n').strip().split(" ")
                for word in doc:
                    stripword = word.strip().strip('\n')
                    classWordSet.add(stripword)
                    docWordSet.add(stripword)

                classDocList.append(docWordSet)

        classDocDic[l[0:7]] = classDocList
        classWordDic[l[0:7]] = classWordSet

    return classDocDic, classWordDic


    

# 对得到的两个词典进行计算，可以得到a b c d 值
# K 为每个类别选取的特征个数
    

def featureSelection(classDocDic, classWordDic, K):
    wordCountDic = dict()
    #the dict of the chi value for each word
    for key in classWordDic:
        classWordSets = classWordDic[key]
        classWordCountDic = dict()
        for eachword in classWordSets:  # 对某个类别下的每一个单词的 a b c d 进行计算
            #print eachword
            a = 0
            b = 0
            c = 0
            d = 0
            for eachclass in classDocDic:
                if eachclass == key:
                    #a, c
                    for eachdoc in classDocDic[eachclass]:
                        if eachword in eachdoc:
                            a += 1
                        else:
                            c += 1
                else:
                    #b, d
                    for eachdoc in classDocDic[eachclass]:
                        if eachword in eachdoc:
                            b += 1
                        else:
                            d += 1
            #print (str(a) + " "+str(c)+" "+str(b)+" "+str(d))
            #print("a+c:"+str(a+c)+"b+d"+str(b+d))
            classWordCountDic[eachword] = Chi(a, b, c, d)
        sortedClassWordCountDic = sorted(classWordCountDic.items(), key = lambda d:d[1], reverse = True)
        #print sortedClassWordCountDic
        tmp = dict()
        for i in range(K):
            tmp[sortedClassWordCountDic[i][0]] = sortedClassWordCountDic[i][1]
        wordCountDic[key] = tmp
    return wordCountDic
        # print(sortedClassTermCountDic)

# 调用buildItemSets
# buildItemSets形参表示每个类别的文档数目,在这里训练模型时每个类别取前200个文件


classDocDic, classWordDic = buildSets()
wordCountDic = featureSelection(classDocDic, classWordDic, 124)

results = set()
for eachclass in wordCountDic:
    for word in wordCountDic[eachclass]:
        results.add(word)

with open("knn/model/KNNFeature.txt", 'w') as f:
    count = 1
    for result in results:
        final_result = result.strip('\n').strip(' ')
        if len(final_result) > 0 and result != " ":
            f.write(str(count) + ' ' + final_result + '\n')
            count = count + 1















