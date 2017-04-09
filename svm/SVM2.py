# -- coding: utf-8 --
from numpy import *  
import time
import random
import matplotlib.pyplot as plt


def kValue(i, j, dataSet):
	result = 0.0
	for k in range(len(dataSet[0])):
		result += float(dataSet[i][k]) * float(dataSet[j][k])
	return result


def Kernel(dataSet, kernelOption):  
    numSamples = len(dataSet)  
    kernel = [[0 for i in range(numSamples)] for j in range(numSamples)]  
    for i in range(numSamples):  
        for j in range(numSamples):
        	kernel[i][j] = kValue(i, j, dataSet)
    return kernel



class SVM:  
    def __init__(self, dataSet, labelSet, C, toler, kernelOption):  
        self.dataSet = dataSet # each row stands for a sample  
        self.labelSet = labelSet  # corresponding label  
        self.C = C             # slack variable  
        self.toler = toler     # termination condition for iteration  
        self.num = len(dataSet) # number of samples  
        self.alphas = [0 for i in range(self.num)] # Lagrange factors for all samples  
        self.b = 0  
        self.errorCache = [[0, 0] for i in range(self.num)]
        self.errorSet = set()
        self.kernelOpt = kernelOption  
        self.kernel = Kernel(self.dataSet, self.kernelOpt)  


def getE(svm, i):
	fi = 0
	for k in range(len(svm.dataSet)):
		fi += float(svm.alphas[k]) * float(svm.labelSet[k]) * float(svm.kernel[k][i])
	fi += svm.b
	Ei = fi - float(svm.labelSet[i])
	return Ei

# main training procedure  

def updateE(svm, i):
	Ei = getE(svm, i)
	svm.errorCache[i][0] = 1
	svm.errorCache[i][1] = Ei
	svm.errorSet.add(i)

def selectJ(svm, i, Ei):
	svm.errorCache[i][0] = 1
	svm.errorCache[i][1] = Ei
	svm.errorSet.add(i)

	j = 0
	Ej = 0

	if len(svm.errorSet) > 1:
		maxValue = 0
		for a in svm.errorSet:
			Ea = getE(svm, a)
			if abs(Ei - Ea) > maxValue:
				maxValue = abs(Ei - Ea)
				j = a
				Ej = Ea
	else:
		j = i
		while j == i:
			j = random.randint(0, len(svm.dataSet) - 1)
		Ej = getE(svm, j)
	return j, Ej


def innerLoop(svm, i):
    Ei = getE(svm, i)
    print i
	### check and pick up the alpha who violates the KKT condition  
    ## satisfy KKT condition  
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
    ## violate KKT condition  
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
    if (svm.labelSet[i] * Ei < -svm.toler) and (svm.alphas[i] < svm.C) or (svm.labelSet[i] * Ei > svm.toler) and (svm.alphas[i] > 0):
		
		#1. select j
        j, Ej = selectJ(svm, i ,Ei)
        oldAi = svm.alphas[i]
        oldAj = svm.alphas[j]

		#2. boundary L and H
        if svm.labelSet[i] != svm.labelSet[j]:
			L = max(0, svm.alphas[j] - svm.alphas[i])
			H = min(svm.C, svm.C - svm.alphas[i] + svm.alphas[j])
        else:
				L = max(0, svm.alphas[i] + svm.alphas[j] - svm.C)
				H = min(svm.C, svm.alphas[i] + svm.alphas[j])
        if L == H:
            return 0

		#3. eta, the similarity of i and j
        eta = 2.0 * svm.kernel[i][j] - svm.kernel[i][i] - svm.kernel[j][j]
        if eta > 0:
            return 0

		#4. updata j
        svm.alphas[j] -= float(svm.labelSet[j]) * (Ei - Ej) / eta

		#5. clip j
        if svm.alphas[j] > H:
			svm.alphas[j] = H
        if svm.alphas[j] < L:  
            svm.alphas[j] = L

        if abs(oldAj - svm.alphas[j]) < 0.00001:
        	updateE(svm, j)
        	return 0
        #6.updata i
        svm.alphas[i] += svm.labelSet[i] * svm.labelSet[j] * float(oldAj - svm.alphas[j])
        #7. updata b1
        b1 = svm.b - Ei - svm.labelSet[i] * (svm.alphas[i] - oldAi) * svm.kernel[i][i] - svm.labelSet[j] * (svm.alphas[j] - oldAj) * svm.kernel[i][j]
        b2 = svm.b - Ej - svm.labelSet[i] * (svm.alphas[i] - oldAi) * svm.kernel[i][j] - svm.labelSet[j] * (svm.alphas[j] - oldAj) * svm.kernel[j][j]

        if 0 < svm.alphas[i] and svm.alphas[i] < svm.C:
            svm.b = b1
        elif 0 < svm.alphas[j] and svm.alphas[j] < svm.C:
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        updateE(svm, j)
        updateE(svm, i)

        return 1
    else:
        return 0


def trainSVM(dataSet, labelSet, C, toler, maxIter, kernelOption = ('rbf', 1.0)):  
    # calculate training time  
    startTime = time.time()  
  
    # init data struct for svm  
    svm = SVM(dataSet, labelSet, C, toler, kernelOption)  
      
    # start training  
    entireSet = True  
    alphaPairsChanged = 0  
    iterCount = 0  
    # Iteration termination condition:  
    #   Condition 1: reach max iteration  
    #   Condition 2: no alpha changed after going through all samples,  
    #                in other words, all alpha (samples) fit KKT condition
    
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):  
        alphaPairsChanged = 0  
        print iterCount   
        # update alphas over all training examples  
        if entireSet:  
            for i in range(svm.num):  
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
            iterCount += 1  
        # update alphas over examples where alpha is not 0 & not C (not on boundary)  
        else:  
            nonBoundAlphasList = list()
            for j in svm.alphas:
            	if j > 0 and j < svm.C:
            		nonBoundAlphasList.append(svm.alphas.index(j))  
            for i in nonBoundAlphasList:  
                alphaPairsChanged += innerLoop(svm, i)  
            print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  
            iterCount += 1  
  
        # alternate loop over all examples and non-boundary examples  
        if entireSet:  
            entireSet = False  
        elif alphaPairsChanged == 0:  
            entireSet = True  
  
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)  
    return svm

def testSVM(svm, test_data, test_label):  
    numTestSamples = len(test_data) 
    supportVectorsIndex = list()
    for i in range(svm.num):
    	if svm.alphas[i] > 0:
    		supportVectorsIndex.append(i) 
    supportVectors = list()
    for i in supportVectorsIndex:
    	supportVectors.append(svm.dataSet[i])
    supportVectorLabels = list()
    for i in supportVectorsIndex:
    	supportVectorLabels.append(svm.labelSet[i])
    supportVectorAlphas = list()
    for i in supportVectorsIndex:
    	supportVectorAlphas.append(svm.alphas[i])
 
    matchCount = 0  
    for i in range(numTestSamples):
    	predict = 0
    	for k in range(len(supportVectors)):
    		kernelValue = 0
    		for x in range(len(test_data[0])):
    			kernelValue += float(supportVectors[k][x]) * float(test_data[i][x])
    		predict += float(supportVectorAlphas[k]) * float(supportVectorLabels[k]) * kernelValue
    	predict += svm.b  
        if sign(predict) == sign(test_label[i]):  
            matchCount += 1
        if predict > 0:
            print '1'
        else:
            print '0'
    accuracy = float(matchCount) / numTestSamples
    return accuracy  