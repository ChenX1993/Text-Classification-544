# -- coding: utf-8 --
import codecs
import math
import sys
import os

classCodes = ['C000013','C000024']
file_path = sys.path[0] + "/../Data"
output_path = 'nb/model/nb_model.txt'

# Stop words
def isStopWord(word):
    with open ('stopwords.txt', 'r') as f:
        words = f.readlines()
        if word in words:
            return True
        else:
            return False



classNum = [0 for i in range(2)]
wordNum =[0 for i in range(2)]
wordSet = dict()

for eachclass in classCodes:
	path = file_path + '/' + eachclass + '_train.txt'
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if line == '\n':
				continue
			classNum[classCodes.index(eachclass)] += 1
			words = line.strip('\n').strip().split(' ')
			for word in words:
				wordNum[classCodes.index(eachclass)] += 1
				if word in wordSet:
					wordSet[word][classCodes.index(eachclass)] += 1
				else:
					wordSet[word] = [0 for i in range(2)]
					wordSet[word][classCodes.index(eachclass)] += 1

#classNum
#wordNum
#wordSet
with open(output_path, 'w') as f:
	wordLen = len(wordSet)
	f.write('SUM ' + str(math.log10(classNum[0]) - math.log10(classNum[0] + classNum[1])) + ' ' + str(math.log10(classNum[1]) - math.log10(classNum[0] + classNum[1])) + '\n')
	for word in wordSet:
		f.write(word + ' ')
		f.write(str(math.log10(wordSet[word][0] + 1) - math.log10(wordNum[0] + wordLen)) + ' ')
		f.write(str(math.log10(wordSet[word][1] + 1) - math.log10(wordNum[1] + wordLen)) + '\n')

