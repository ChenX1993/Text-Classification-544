# -- coding: utf-8 --
#!/usr/bin/python
import codecs
import math
import os
import numpy as np
from sklearn.decomposition import PCA

# Cosin similarity
def sim(test_dic, train_dic, dicFeature):  #test_dic--样本列表
    b = 0.0 
    c1 = 0.0  
    c2 = 0.0
    sim = 0.0
    for index in test_dic: 
        x = float(test_dic[index])
        if index not in train_dic:
            y = 0.0
        else:
            y = float(train_dic[index]) 
        b += x * y
        c1 += math.pow(x, 2)  

    for index in train_dic:    
        c2 += math.pow(float(train_dic[index]), 2) 

    if (math.sqrt(c1) * math.sqrt(c2) == 0):
        sim = 0.0
    else:
        sim = b / (math.sqrt(c1) * math.sqrt(c2)) #cos值
    return sim

# def sim1(test_dic, train_dic, dicFeature):  #test_dic--样本列表
#     b = 0.0 
#     c1 = 0.0  
#     c2 = 0.0
#     sim = 0.0

#     for index in dicFeature:
#         if index in test_dic:
#             x = float(test_dic[index])
#         else:
#             x = 0.0
#         if index not in train_dic:
#             y = 0.0
#         else:
#             y = float(train_dic[index]) 

#         b += x * y
#         c1 += math.pow(x, 2)  
#         c2 += math.pow(y, 2) 

#     # for index in train_dic:    
#     #     c2 += math.pow(float(train_dic[index]), 2) 

#     if (math.sqrt(c1) * math.sqrt(c2) == 0):
#         sim = 0.0
#     else:
#         sim = b / (math.sqrt(c1) * math.sqrt(c2)) #cos值

#     return sim
# Euclidean distance

# def ecli(test_dic, train_dic, dicFeature):
#     c = 0.0
#     d = 0.0
#     x = list()
#     y = list()
#     xy_list = list()
   
#     # for index in dicFeature:
#     #     if index not in test_dic:
#     #         x.append(0.0)
#     #     else:
#     #         x.append(float(test_dic[index]))
#     #     if index not in train_dic:
#     #         y.append(0.0)
#     #     else:
#     #         y.append(float(train_dic[index]))

#     xy_list.append(x)
#     xy_list.append(y)
# #    print xy_list
#     for 
#     pca = PCA(n_components = 3)
#     xList = pca.fit_transform()
#     xy_newList = pca.fit_transform(xy_list)
#     for e in xy_newList:
#         c += math.pow((xy_newList[0][0] - xy_newList[1][0]),2) + math.pow((xy_newList[0][1] - xy_newList[1][1]),2)

#     d = math.sqrt(c)
# #    print d
#     return d

def classify(k, docWeight, testWeight,dicFeature):
     # print docWeight
    result = list()
    for key in testWeight:
        li = list()
        test_lable = testWeight[key][0]#待分类文档类别
        test_dic = testWeight[key][1]
      
        for doc_key in docWeight: 

            train_lable = docWeight[doc_key][0]

            train_dic = docWeight[doc_key][1]
            # print test_dic
            # s = sim(test_dic, train_dic)
            s = sim(test_dic, train_dic, dicFeature)
            # if (int(measure) == 2):
            #     s = sim(test_dic, train_dic, dicFeature)
            # elif (int(measure) == 1):
            #     s = ecli(test_dic, train_dic, dicFeature)
            li.append((s,test_lable,train_lable))
        
        li.sort(reverse = True) #排序
        # print li
        tmpli = li[0:int(k)]         #取前k个
#        print tmpli
        di = dict()
        for l in tmpli: #遍历K个邻居
            s,test_lable,train_lable = l 
            if (int(train_lable)) in di:
                di[int(train_lable)] += 1
            else:
                di[int(train_lable)] = 1
            
        sortDi = sorted(di.items(),None, key=lambda d: d[1], reverse = True) #排序取文档数最多的类
#        print sortDi
        result_lable = sortDi[0][0]
        # print di
        result.append((int(result_lable), int(test_lable)))
        
    return result


def main():
    print "-Step 5: testing..."
    file_path = "knn/model/train.svm"
    fileText_path = "knn/model/test.svm"

    weightDoc = list()
    docWeight = dict()

    weigthTest = list()  
    testWeight = dict()

    dicFeature = dict()

    docId = 1
    with open("knn/model/KNNFeature.txt", "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            feature = line.strip().split(' ')
            dicFeature[feature[0]] = feature[1]
        # print dicFeature
    length = len(dicFeature)
    Class = dict()
    with open(file_path, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:    
            # print line
            feature = list()
            dicDoc = dict()
            weightDoc = line.strip().split(' ')
            if (len(weightDoc) > 1):
                feature.append(weightDoc[0]) #weight[0] -- label;
                if weightDoc[0] not in Class:
                    Class[weightDoc[0]] = 1
                else:
                    Class[weightDoc[0]] += 1

                for i in range(1,len(weightDoc)): #weight[1:] -- f_id:idf
                    if (weightDoc[i] != ''):
                        # print weightDoc[i]
                        element = weightDoc[i].strip('\n').split(':') #element -- [f_id, idf]
                        # print element
                        dicDoc[element[0]] = element[1]
                        # print element[0]

                feature.append(dicDoc)
                docWeight[docId] = feature  # docWeight{id : [lable, {f_id : idf}]}
                docId += 1
    countClass = len(Class)
    print countClass

    testId = 1
    with open(fileText_path, "r") as f_in:
        lines = f_in.readlines()
        testLen = 0
        for line in lines:
            feature = list()
            dicTest = dict()
            weigthTest = line.strip('\n').split(' ')

            if (len(weigthTest) > 1):
                feature.append(weigthTest[0]) #weight[0] -- label;
                for i in range(1,len(weigthTest)): #weight[1:] -- f_id:idf
                    if (weigthTest[i] != ''):
                        element = weigthTest[i].strip('\n').split(':') #element -- [f_id, idf]
                        dicTest[element[0]] = element[1]

                feature.append(dicTest)
                testWeight[testId] = feature  # testWeight{id : [lable, {f_id : idf}]}

                testId += 1
    # print testWeight

    # while True:
    measure = 0
    # print '****************'
    # print 'Choose one of training meathods:'
    # inputStr = raw_input('[1] Euclidian distance measure; [2] Cosine distance measure; [3] Quit KNN: ')
    # if (inputStr == '1'):
    #     measure = 1
    # elif (inputStr == '2'):
    #     measure = 2
    # elif (inputStr == '3'):
    #     breakå
    # else:
    #     print '*Warning! Your input is invalid. Please enter a correct input.'
    k = math.floor(math.sqrt(docId -1))


    tupleReuslt = classify(k, docWeight, testWeight, dicFeature)
    accur = calculate(k,tupleReuslt)
    print "Accuracy:", accur, "K value:", k
    f_micro = fScore(tupleReuslt, countClass)
    # print "F1:", F, "K value:", k

    # if (measure == 2):
    with open("result/knn_result_cosin.txt",'w') as output:
        output.write(str(round(accur,6)) +'\n')
        for e in tupleReuslt:
            output.write(str(e[0])+'\n')
        # if (measure == 1):
        #     with open("result/knn_result_ecli.txt",'w') as output:
        #         output.write(str(round(F,6)) +'\n')
        #         for e in tupleReuslt:
        #             output.write(str(e[0])+'\n')

    # while True:
    #    print '****************'
    #    inputStr = raw_input('Try other value of k.(Y/N)')
    #    if inputStr == 'Y':
    #        k = raw_input('Please input k value (Integer only):')
    #        tupleReuslt = classify(int(k), docWeight, testWeight, dicFeature)
    #        calculate(int(k),tupleReuslt)
    #    elif inputStr == 'N':
    #        break
    #    else:
    #        print '*Warning! Your input is invalid. Please enter a correct input.'
    #        continue
def fScore(tupleReuslt, numOfClasses):
    r_micro = 0.0
    p_micro = 0.0
    r_macro = 0.0
    p_macro = 0.0

    tp = list()
    fp = list()
    fn = list()
    # get tp, fp, fn value for each class

    for classIndex in range(numOfClasses):
        tpNum = 0
        fpNum = 0
        fnNum = 0
        for e in tupleReuslt:
            if e[0] == classIndex:
                if e[1] == classIndex:
                    tpNum += 1
                else:
                    fpNum += 1
            else:
                if e[1] == classIndex:
                    fnNum += 1
        tp.append(tpNum)
        fp.append(fpNum)
        fn.append(fnNum)

    #micro p, r
    pN = 0.0
    pD = 0.0
    rN = 0.0
    rD = 0.0
    for i in range(numOfClasses):
        pN += tp[i]
        pD += tp[i] + fp[i]
        rN += tp[i]
        rD += tp[i] + fn[i]
    p_micro = float(pN) / pD
    r_micro = float(rN) / rD

    # #macro p, r

    # for i in range(numOfClasses):
    #   p = float(tp[i]) / (tp[i] + fp[i])
    #   r = float(tp[i]) / (tp[i] + fn[i])
    #   p_macro += p
    #   r_macro += r
    # p_macro = p_macro / numOfClasses
    # r_macro = r_macro / numOfClasses

    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro)
    #f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)

    print ('F1_micro score:' + str(round(f_micro, 6)))
    #print ('F1_macro score:' + str(round(f_macro, 6)))

    return round(f_micro, 6)
def calculate(k,tupleReuslt):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    #属于类1的样本被正确分类到类1，TP
    length = len(tupleReuslt)
    numAbove = 0
    for e in tupleReuslt:
        if (int(e[0]) == int(e[1])) :
            numAbove += 1
    accur = (numAbove + 0.0) / (length + 0.0)
        # if(int(e[1]) == 1 and int(e[0]) == 1):
        #     TP += 1
        # #不属于类1的样本被错误分类到类1, FN
        # if(int(e[1]) != 1 and int(e[0]) == 1):
        #     FN += 1
        # #属于类别1的样本被错误分类到类0, TN
        # if(int(e[1]) == 1 and int(e[0]) != 1):
        #     TN += 1
        # #不属于类别C的样本被正确分类到了类别C的其他类  FP
        # if(int(e[1]) != 1 and int(e[0]) != 1):
        #     FP += 1
    
    # precision = (TP + 0.0) / (TP + FN + 0.0)
    # recall = (TP + 0.0) / (TP + TN + 0.0)
    # F = (2 * precision * recall) / (precision + recall + 0.0)
    


    return accur




main()
