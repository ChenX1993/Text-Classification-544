# -- coding: utf-8 --
# import FeatureSelecion

import math
import sys
import os


# 采用TF-IDF 算法对选取得到的特征进行计算权重
#the nums of docs selected for each class
DocumentCount = 50 

#ClassCodes = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']
ClassCodes = ['C000013','C000024']

textCutPath = sys.path[0] + "/../Data"
featurePath = 'svm/model/SVMFeature.txt'
dffeaturePath = 'svm/model/dffeature.txt'
tfidfPath = 'svm/model/train.svm'


# 计算特征的逆文档频率
def featureIDF(dic, features):
    #dic key: class   value: word
    #features feature list
    f = open(dffeaturePath, "w")
    f.close()
    f = open(dffeaturePath, "a")

    docNum = 0

    idffeature = dict()

    for feature in features:
        featureDocNum = 0
        for eachclass in dic:        
            docNum += len(dic[eachclass])
            docList = dic[eachclass]    
            for words in docList:        #期中一个file
                if feature in words:  #如果当前特征在这个file里面，docfeature ++ feature出现频率
                    featureDocNum += 1
        # 计算特征的逆文档频率
        featurevalue = math.log(float(docNum)/(featureDocNum+1))
        # 写入文件，特征的文档频率
        f.write(feature + " " + str(featureDocNum)+"\n")
        #print(eachfeature+" "+str(docfeature))
        idffeature[feature] = featurevalue
    f.close()

    return idffeature
    #return the value of idf

# 计算Feature's TF-IDF 值
def TFIDFCal(features, dic,idffeature):
    f = open(tfidfPath, 'w')
    f.close()
    f = open(tfidfPath, 'a')

    for eachclass in dic:
        classIndex = ClassCodes.index(eachclass)
        for doc in dic[eachclass]:
            f.write(str(classIndex)+" ")
            for feature in features:
                #print features[i]
                if feature in doc:
                    featureNum = doc.count(feature)
                    tf = float(featureNum)/(len(doc))
                    # 计算逆文档频率
                    featurevalue = tf * idffeature[feature]
                    f.write(str(features.index(feature) + 1)+":"+str(featurevalue) + " ")
            f.write("\n")
    f.close()

#dic = readFileToList(textCutPath, ClassCode, DocumentCount)

# get features from the txt
features = list()
with open(featurePath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        feature = line.split(' ')[1].strip('\n')
        #print feature
        features.append(feature)

# read file to list
dic = dict()

for eachclass in ClassCodes:
    path = textCutPath + '/' + eachclass + '_train.txt'
    docList = list()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc = line.split(' ')
            docList.append(doc)
    dic[eachclass] = docList

# get the list of idf
idffeature = featureIDF(dic, features)
#get tf - idf for features in each class
TFIDFCal(features, dic,idffeature)











