# -- coding: utf-8 --
  
from numpy import *  
import SVM
import SVM3


## step 1: load data  
# def scale(dataSet):
# 	maxValue = 0
# 	for i in range(len(dataSet)):
# 		for j in range(len(dataSet[0])):
# 			if abs(dataSet[i][j]) > maxValue:
# 				maxValue = abs(dataSet[i][j])
# 	for i in range(len(dataSet)):
# 		for j in range(len(dataSet[0])):

# 			dataSet[i][j] = dataSet[i][j] / maxValue
# 	return dataSet

def scale(dataSet):
	for j in range(len(dataSet[0])):
		maxValue = 0
		for i in range(len(dataSet)):
			if abs(dataSet[i][j]) > maxValue:
				maxValue = abs(dataSet[i][j])
		if maxValue != 0:
			for i in range(len(dataSet)):
				dataSet[i][j] = dataSet[i][j] / maxValue

# 			dataSet[i][j] = dataSet[i][j] / maxValue 
# 			dataSet[i][j] = dataSet[i][j] / maxValue
# 	return dataSet

def scale(dataSet):
	maxValue = 0
	for i in range(len(dataSet)):
		if abs(dataSet[i][0]) > maxValue:
			maxValue = abs(dataSet[i][0])
	for i in range(len(dataSet)):
		dataSet[i][0] = dataSet[i][0]/maxValue
	maxValue = 0
	for i in range(len(dataSet)):
		if abs(dataSet[i][1]) > maxValue:
			maxValue = abs(dataSet[i][1])
	for i in range(len(dataSet)):
		dataSet[i][1] = dataSet[i][1]/maxValue

	return dataSet


print "----- 5.1: load data..."

trainDataSet = list()  
trainLabelSet = list()
testDataSet = list()
testLabelSet = list()
	
with open('svm/model/new_train.svm', 'r') as f:
	lines = f.readlines()
	for line in lines:
		content = line.strip().split(' ')
		# trainDataSet.append([float(content[0]), float(content[1])])
		# trainLabelSet.append(float(content[2]))
		tmpDataSet = list()
		for i in range(len(content)):
			if i == 0:
				trainLabelSet.append(float(content[0]))
			else:
				tmpDataSet.append(float(content[i]))
		trainDataSet.append(tmpDataSet)

with open('svm/model/new_test.svm', 'r') as f:
	lines = f.readlines()
	for line in lines:
		content = line.strip().split(' ')
		# testDataSet.append([float(content[0]), float(content[1])])
		# testLabelSet.append(float(content[2]))
		tmpDataSet = list()
		for i in range(len(content)):
			if i ==0:
				testLabelSet.append(float(content[0]))
			else:
				tmpDataSet.append(float(content[i]))
		testDataSet.append(tmpDataSet)

trainDataSet = scale(trainDataSet)
testDataSet = scale(testDataSet)


trainDataSet = mat(trainDataSet)
trainLabelSet = mat(trainLabelSet).T

testDataSet = mat(testDataSet)
testLabelSet = mat(testLabelSet).T  

  
# step 2: training...  
print "----- 5.2: training..."
C = 0.8
toler = 0.0001
maxIter = 30
svmClassifier = SVM3.trainSVM(trainDataSet, trainLabelSet, C, toler, maxIter, kernelOption = ('rbf', 0))  
  
# step 3: testing  
print "----- 5.3: testing..."
accuracy = SVM3.testSVM(svmClassifier, testDataSet, testLabelSet)  
  
# step 4: show the result  
#print "----- 5.4: show the result..."    
#print 'Accuracy: %.3f' % (accuracy)  
#SVM.showSVM(svmClassifier) 
