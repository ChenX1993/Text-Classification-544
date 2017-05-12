# -- coding: utf-8 --
import math
import sys
import os

#class id list

textCutPath = sys.path[0] + "/../Data/dev"
featurePath = 'knn/model/KNNFeature.txt'
dffeaturePath = 'knn/model/dffeature.txt'
tfidfPath = 'knn/model/test.svm'

TestDocumentCount = 20
DocumentCount = 200
TrainDocumentCount = 2000



# 读取特征的文档计数  特征词频
def readDfFeature(dffilename):
    dffeaturedic = dict()
    dffile = open(dffilename, "r")
    dffilecontent = dffile.read().split("\n")
    dffile.close()
    for eachline in dffilecontent:
        eachline = eachline.split(" ")
        if len(eachline) == 2:
            dffeaturedic[eachline[0]] = eachline[1]
            # print(eachline[0] + ":"+eachline[1])
    # print(len(dffeaturedic))
    return dffeaturedic

# 对测试集进行特征向量表示
def readFileToList(textCutPath, ClassCodes):
    dic = dict()  #读取各个class 的test 存在同一个dict
    for eachclass in ClassCodes:
        currClassPath = textCutPath + eachclass + "/"
        eachclasslist = list()
        for filename in os.listdir(currClassPath):
            #print filename
            eachfile = open(currClassPath+'/'+filename, 'r')
            eachfilecontent = eachfile.read()
            eachfilewords = eachfilecontent.split(" ")
            eachclasslist.append(eachfilewords)
            # print(eachfilewords)
        dic[eachclass] = eachclasslist
    return dic

def TFIDFCal(features, dic,dffeatures,fList):
    f = open(tfidfPath, 'w')
    f.close()
    f = open(tfidfPath, 'a')

    for eachclass in dic:
        classIndex = fList.index(eachclass)
        for doc in dic[eachclass]:
            f.write(str(classIndex)+" ")
            for feature in features:
                #print features[i]
                if feature in doc:
                    featureNum = doc.count(feature)
                    tf = float(featureNum)/(len(doc))
                    # 计算逆文档频率
                    idffeature = math.log(float(TrainDocumentCount+1)/(int(dffeatures[feature])+2))
                    featurevalue = tf * idffeature
                    f.write(str(features.index(feature) + 1)+":"+str(featurevalue) + " ")
            f.write("\n")
    f.close()

# 对200至250序号的文档作为测试集
#get feature

features = list()
with open(featurePath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        feature = line.split(' ')[1].strip()
        #print feature
        features.append(feature)
#get df 
dffeatures = dict()
with open(dffeaturePath, 'r') as f:
    lines = f.read().split("\n")
    for line in lines:
        line = line.split(" ")
        if len(line) == 2:
            dffeatures[line[0]] = line[1]

# read file to list
dic = dict()
fList = list()
if os.path.isdir(textCutPath):
    files = os.listdir(textCutPath)
    for f in files:
        # print f
        if f != ".DS_Store":
            fList.append(f[0:7])

for eachclass in fList:
    path = textCutPath + '/' + eachclass + '_dev.txt'
    docList = list()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc = line.split(' ')
            docList.append(doc)
    dic[eachclass] = docList

TFIDFCal(features, dic, dffeatures, fList)
