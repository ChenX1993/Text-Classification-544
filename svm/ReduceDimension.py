# -- coding: utf-8 --
import numpy 
import array
import re
from sklearn.decomposition import PCA  

modelPath = 'svm/model/train.svm'
testPath = 'svm/model/test.svm'
featurePath = 'svm/model/SVMFeature.txt'
newModelPath = 'svm/model/new_train.svm'
newTestPath = 'svm/model/new_test.svm'


def scale(testDic):
	denominator = - 1.0
	for key in testDic:
		if testDic[key] > denominator:
			denominator = testDic[key]
	for key in testDic:
		testDic[key] =testDic[key] / denominator
	return testDic

def readFileToList(featureLen, filePath):
	docList = list()
	classIdList = list()
	with open(filePath, 'r') as f:
		lines = f.readlines()
		for line in lines:
			tmp = [0 for i in range(featureLen)]
			features = dict()
			line = line.strip('\n').strip(' ').split(' ')
			if (int(line[0]) == 0):
				classIdList.append(-1)
			else:
				classIdList.append(1)
			#tmp.append([0 i for range(featureLen)])
			for feature in line[1:]:
				m = re.match(r'^(.*):(.*)$', feature)
				features[int(m.group(1))] = float(m.group(2))
				#features = scale(features)
				for key in features:
					tmp[key - 1] = features[key]
			docList.append(tmp)
	return docList, classIdList

def getFeatureLen():
	with open(featurePath, 'r') as f:
		lines = f.readlines()
		return len(lines)

def reduceD(docList):
	newDocList = list()
	pca = PCA(n_components = 50)
	newDocList = pca.fit_transform(docList)
	return newDocList

def writeListToFile(newDocList, classIdList, filePath):
	with open(filePath, 'w') as f:
		for i in range(len(classIdList)):
			# f.write(str(newDocList[i][0]) + ' ' + str(newDocList[i][1]) + ' ' + str(classIdList[i]))
			f.write(str(classIdList[i]))
			for j in range(len(newDocList[0])):
				f.write(' ' + str(newDocList[i][j]))
			f.write("\n")

#reduce the training data to 2D
print('----- 4.1: reduce the training data to 50D...')
featureLen = getFeatureLen()
docList = list()
classIdList = list()
docList, classIdList = readFileToList(featureLen, modelPath)
newDocList = list()
newDocList = reduceD(docList)
#print len(newDocList)
#print len(classIdList)
writeListToFile(newDocList, classIdList, newModelPath)
print('----- 4.2: training data complete')

#reduce the test data to 2D
print('----- 4.3: reduce the test data to 50D...')
docList, classIdList = readFileToList(featureLen, testPath)
newDocList = reduceD(docList)
writeListToFile(newDocList, classIdList, newTestPath)
print('----- 4.4: test data complete')
