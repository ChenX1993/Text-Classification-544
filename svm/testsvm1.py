# -- coding: utf-8 --
  
from numpy import *  
import SVM  
import random

C = 1.0
tal = 0.01
maxPasses = 5
trainDataSet = list()
trainLabelSet = list()
with open('model/new_train.svm', 'r') as f:
	lines = f.readlines()
	for line in lines:
		content = line.strip().split(' ')
		trainDataSet.append([float(content[0]), float(content[1])])
		trainLabelSet.append(float(content[2]))

trainDataSet = trainDataSet[0:101]
trainLabelSet = trainLabelSet[0:101]

def initiateKernel():
	for i in range(len(trainDataSet)):
		for j in range(len(trainDataSet)):
			kernel[i][j] = kValue(i, j)

def kValue(i, j):
	result = 0
	for k in range(len(trainDataSet[0])):
		result += trainDataSet[i][k] * trainDataSet[j][k]
	return result

def f(j):
	result = 0
	for i in range(len(trainDataSet)):
		result += a[i] * trainLabelSet[i] * kernel[i][j]
	return result + b

def getE(i):
	result = f(i) - trainLabelSet[i]
	return result

def findMax(Ei):
	maxValue = 0
	maxIndex = -1
	for a in boundAlpha:
		Ej = getE(a)
		if abs(Ei - Ej) > maxValue:
			maxValue = abs(Ei - Ej)
			maxIndex = a
	return maxIndex

a = [0 for i in range(len(trainDataSet))]
passes = 0
b = 0
#kernel = [[0 for i in range(len(trainDataSet))] for j in range(len(trainDataSet))]
kernel = list()
for i in range(len(trainDataSet)):
	tmp = list()
	for j in range(len(trainDataSet)):
		tmp.append(0)
	kernel.append(tmp)
initiateKernel()
boundAlpha = set()

count = 0
while passes < maxPasses:
	print 'count: ' + str(count)
	print 'passes: ' + str(passes)
	count += 1
	num_alphas_changed = 0
	for i in range(len(trainDataSet)):
		Ei = getE(i)
		if (trainLabelSet[i] * Ei < - tal and a[i] < C) or (trainLabelSet[i] * Ei > tol and a[i] > 0):
			#print 'checkpont1'
			j = 0
			if len(boundAlpha) > 0:
				j = findMax(Ei)
			else:
				j = random.randint(0, len(trainDataSet) - 1)
				while (j == i):
					j = random.randint(0, len(trainDataSet) - 1)
			print j
			Ej = getE(j)
			oldAi = float(a[i])
			oldAj = float(a[j])
			L = 0
			H = 0
			print Ej
			print oldAj
			print oldAi 

			if trainLabelSet[i] != trainLabelSet[j]:
				L = max(0, a[j] - a[i])
				H = min(C, C - a[i] + a[j])
			else:
				L = max(0, a[i] + a[j] - C)
				H = min(C, a[i] + a[j])

			print 'L' + str(L)
			print 'H' + str(H)
			eta = 2 * kValue(i, j) - kValue(i, i) - kValue(j, j)
			#print 'checkpont2'
			if eta >= 0:
				continue

			a[j] = a[j] - trainLabelSet[j] * (Ei - Ej) / eta
			if 0 < a[j] and a[j] < C:
				boundAlpha.add(j)

			if a[j] < L:
				a[j] = L
			elif a[j] > H:
				a[j] = H
			#print 'checkpont3'
			#print abs(a[j]-oldAj)
			if abs(a[j] - oldAj) < 0.00001:
				continue;
			a[i] = a[i] + trainLabelSet[i] * trainLabelSet[j] * (oldAj - a[j])

			if 0 < a[i] and a[i] < C:
				boundAlpha.add(i)
			#print 'checkpont3.1'
			b1 = b - Ei - trainLabelSet[i] * (a[i] - oldAi) * kValue(i, i) - trainLabelSet[j] * (a[j] - oldAj) * kValue(i, j)
			b2 = b - Ej - trainLabelSet[i] * (a[i] - oldAi) * kValue(i, j) - trainLabelSet[j] * (a[j] - oldAj) * kValue(j, j)
			#print 'checkpont3.5'
			if 0 < a[i] and  a[i] < C:
				b = b1
			elif 0 < a[j] and a[j] < C:
				b = b2
			else:
				b = (b1 + b2) / 2
			#print 'checkpont4'
			num_alphas_changed += 1
			print num_alphas_changed

		if num_alphas_changed ==0:
			passes += 1
		else:
			passes = 0


