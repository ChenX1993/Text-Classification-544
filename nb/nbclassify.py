import math
import os
import sys
classCodes = ['C000013','C000024']
file_path = sys.path[0] + "/../Data"
model_path = 'nb/model/nb_model.txt'
#output_path = 'nb/model/nb_result.txt'
output_path = 'result/nb_result.txt'

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
	i = 1
	for eachclass in classCodes:
		path = file_path + '/' + eachclass + '_dev.txt'
		with open(path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				if line == '\n': 
					continue

				#trueLabel.append(str(classCodes.index(eachclass)))
				trueLabel.append(str(i))
				words = line.strip('\n').strip().split(' ')
				pc0 = pCal(classP, wordSet, words, 0)
				pc1 = pCal(classP, wordSet, words, 1)
				if pc0 > pc1:
					label.append('1')
				else:
					label.append('0')
		i -= 1
		
	return label, trueLabel


def pCal(classP, wordSet, words, c):
	p = classP[c]
	for word in words:
		if word not in wordSet:
			continue
		else:
			p += wordSet[word][c]
	return p

import re
import numpy as np 
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pa 


classes_dic = []
allwords = {}
prior = []
test = []
correct_label = []
predict_label = []
raw_data = []
lines = list(codecs.open("nb/model/nbmodel.txt", "r", "utf-8").readlines())
num_classes = int(lines[0].strip())
pre_data = num_classes+1
for i in range(1, num_classes+1):
	prior.append(lines[i].strip())
	classes_dic.append({})
num_word = (len(lines)-pre_data)/pre_data
for i in range(0, num_word):
	word = lines[pre_data+i*pre_data].strip('\n')
	# print word
	for j in range(0, num_classes):
		p = float(lines[pre_data+i*pre_data+j+1].strip('\n'))
		classes_dic[j][word] = p

test_num = 0
input_path = 'Data/dev'
i = 0
for filename in os.listdir(input_path):
	if re.match(r'^C\d{5}', filename):
		one_file = list(codecs.open(input_path+'/'+filename, "r", "utf-8").readlines())
		for line in one_file:
			line = line.strip()
			raw_data.append(line)
			word_list = line.split(" ")
			test.append(word_list)
			test_num += 1
			correct_label.append(i)
		i += 1
for i in range(0, len(prior)):
	prior[i] = math.log(float(prior[i]), 10)

score = []
for one_test in test:
	for i in range(0, num_classes):
		one_score = prior[i]
		for word in one_test:
			if word in classes_dic[i]:
				one_score += math.log(float(classes_dic[i][word]), 10)

		score.append(one_score)
	max_class = 0
	max_p = score[0]
	for i in range(1, num_classes):
		if score[i]>max_p:
			max_p = score[i]
			max_class = i
	predict_label.append(max_class)
	score = []
#correct_predict = 0
# for i in range(0, len(correct_label)):
# 	real_label = correct_label[i]
# 	pre_label = predict_label[i]
# 	if (real_label == pre_label):
# 		correct_predict += 1
# accuracy = float(correct_predict)/float(test_num)

output_path = "result/nb_result.txt"

def writeResultToFile(label,acc):
	with open(output_path, 'w') as f:
		f.write(str(acc)+'\n')
		for eachlabel in label:
			f.write(str(eachlabel) + '\n')

def accuracy(label, trueLabel):
	correct_predict = 0
	for i in range(0, len(correct_label)):
		real_label = correct_label[i]
		pre_label = predict_label[i]
		if (real_label == pre_label):
			correct_predict += 1
	accuracy = float(correct_predict)/float(test_num)
	print "\nPredict Results:"
	print "Total number of test cases : " + str(test_num)
	print "Total number of correct predictions : " + str(correct_predict)
	print 'Accuracy: ' + str(round(accuracy,6)) +'\n'
	return str(round(accuracy,6))
def fScore(resultLabelSet, trueLabelSet, numOfClasses):
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
		for i in range(len(resultLabelSet)):
			if resultLabelSet[i] == classIndex:
				if trueLabelSet[i] == classIndex:
					tpNum += 1
				else:
					fpNum += 1
			else:
				if trueLabelSet[i] == classIndex:
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

	#macro p, r

	for i in range(numOfClasses):
		p = float(tp[i]) / (tp[i] + fp[i])
		r = float(tp[i]) / (tp[i] + fn[i])
		p_macro += p
		r_macro += r
	p_macro = p_macro / numOfClasses
	r_macro = r_macro / numOfClasses

	f_micro = 2 * p_micro * r_micro / (p_micro + r_micro)
	f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)

	print ('F1_micro score:' + str(round(f_micro, 6)))
	print ('F1_macro score:' + str(round(f_macro, 6)))

	return round(f_micro, 6), round(f_macro, 6)
acc = accuracy(predict_label, correct_label)
f_micro, f_macro = fScore(predict_label, correct_label, num_classes)
writeResultToFile(predict_label, f_micro)









