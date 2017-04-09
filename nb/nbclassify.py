# -- coding: utf-8 --
import codecs
import math
import sys
import os

classCodes = ['C000013','C000024']
file_path = sys.path[0] + "/../Data"
model_path = 'nb/model/nb_model.txt'
output_path = 'nb/model/nb_result.txt'


def readModeToList():
	classP = list()
	wordSet = dict()

	with open(model_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			words = line.strip('\n').strip().split(' ')
			if words[0] == 'SUM':
				classP.append(float(words[1]))
				classP.append(float(words[2]))
			else:
				if words[0] not in wordSet:
					wordSet[words[0]] = [0 for i in range(2)]
				wordSet[words[0]][0] = float(words[1])
				wordSet[words[0]][1] = float(words[2])
	return wordSet, classP

def testDataCal(classP, wordSet):
	label = list()
	trueLabel = list()

	#test data
	for eachclass in classCodes:
		path = file_path + '/' + eachclass + '_dev.txt'
		with open(path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				if line == '\n':
					continue

				trueLabel.append(str(classCodes.index(eachclass)))
				words = line.strip('\n').strip().split(' ')
				pc0 = pCal(classP, wordSet, words, 0)
				pc1 = pCal(classP, wordSet, words, 1)
				if pc0 > pc1:
					label.append('0')
				else:
					label.append('1') 
	return label, trueLabel

def pCal(classP, wordSet, words, c):
	p = classP[c]
	for word in words:
		if word not in wordSet:
			continue
		else:
			p += wordSet[word][c]
	return p

def writeResultToFile(label):
	with open(output_path, 'w') as f:
		for eachlabel in label:
			f.write(eachlabel + '\n')

def accuracy(label, trueLabel):
	length = float(len(label))
	TP = 0
	FP = 0
	FN = 0
	for i in range(len(label)):
		if (label[i] == '1'):
			if trueLabel[i] == '1':
				TP += 1
			else:
				FP += 1
		elif trueLabel == '1':
			FN += 1
	P = float(TP) / (TP + FP)
	R = float(TP) / (TP + FN)
	F1 = 2.0 * P * R / (P + R)
	print 'F1 score: ' + str(F1) 
	#print P
	#print R
	#print 'accuracy: ' + str(F1)


wordSet = dict()
classP = list()

wordSet, classP = readModeToList()

label = list()
trueLabel = list()

label, trueLabel = testDataCal(classP, wordSet)

writeResultToFile(label)
accuracy(label, trueLabel)

