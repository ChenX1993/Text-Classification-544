# -- coding: utf-8 --
import os
import time

def CNN():
	print ''
	print 'CNN'


def KNN():
	print ''
	print '------KNN------'
	print ''

	print'-Step 1: feature selection...'
	os.system('python knn/my_feature_selection.py')

	print '-Step 2: calculate training feature weight...'
	os.system('python knn/my_feature_weight.py')

	print '-Step 3: calculate test feature weight...'
	os.system('python knn/my_test_feature_weight.py')

	print '-Step 4: train model and test data...'
	os.system('python knn/KNN.py')

	print '------KNN complete------'

def SVM():
	print ''
	print '------SVM------'
	print ''

	print'-Step 1: feature selection...'
	os.system('python svm/feature_selection.py')

	print '-Step 2: calculate training feature weight...'
	os.system('python svm/feature_weight.py')

	print '-Step 3: calculate test feature weight...'
	os.system('python svm/test_feature_weight.py')

	print '-Step 4: dimensionality reduction...'
	os.system('python svm/ReduceDimension.py')

	print '-Step 5: train model and test data...'
	os.system('python svm/testsvm.py')

	print '------SVM complete------'




def NB():
	print ''
	print '------Naiev Bayes------'
	print ''

	print '-Step 1: model training...'
	os.system('python nb/nblearn.py')

	print '-Step 2: test data...'
	os.system('python nb/nbclassify.py')
	print '------Naiev Bayes complete------'



print '------Text Classification------'

while True:
	print ''
	print ''
	print '[1] CNN'
	print '[2] KNN'
	print '[3] SVM'
	print '[4] Naive Bayes'
	print '[5] quite'
	print ''

	inputStr = raw_input('Choose you classification method(only number valid): ')

	if inputStr == '1':
		CNN()
	elif inputStr == '2':
		KNN()
	elif inputStr == '3':
		SVM()
	elif inputStr == '4':
		NB()
	elif inputStr == '5':
		break
	else:
		print '*Warning! Your input is invalid. Please enter a correct input.'
		continue