import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import os
import codecs

classes_dics = []
allwords = {}
total_sample = 0
input_path = "Data/train"
num_classes = len([x for x in os.listdir(input_path) if os.path.isfile(input_path+'/'+x) and re.match(r'^C\d{5}', x)])
class_sample = [0]*num_classes
class_size = [0]*num_classes
labels = []

def addToDic(dictionary, word_list):
	for word in word_list:
		if word in dictionary:
			dictionary[word] += 1
		else:
			dictionary[word] = 1

i = 0
for filename in os.listdir(input_path):
	if re.match(r'^C\d{5}', filename):
		classes_dics.append({})
		labels.append(filename[:-10])
		one_file = list(codecs.open(input_path+'/'+filename, "r", "utf-8").readlines())
		for line in one_file:
			total_sample += 1
			line = line.strip()
			word_list = line.split(' ')
			addToDic(classes_dics[i], word_list)
			addToDic(allwords, word_list)
			class_sample[i] += 1
		i += 1
print "\nTotal training sample : " + str(total_sample)
prior = [0]*num_classes
for j in range(0, num_classes):
	prior[j] = float(class_sample[j])/float(total_sample)

def addOne(word):
	for i in range(0, len(classes_dics)):
		dic = classes_dics[i]
		if word in dic:
			dic[word] += 1
		else:
			dic[word] = 1
		class_size[i] += dic[word]
def calculate(dictionary, size):
	for word in dictionary:
		count = dictionary[word]
		dictionary[word] = float(count)/float(size)
def process():
	for word in allwords:
		addOne(word)
	for i in range(0, len(classes_dics)):
		calculate(classes_dics[i], class_size[i])

process()
with open("nb/model/nbmodel.txt",'w') as model:
	model.write(str(num_classes)+'\n')
	for j in range(0, num_classes):
		model.write(str(prior[j])+'\n')
	for word in allwords:
		model.write(word+'\n')
		for j in range(0, len(classes_dics)):
			model.write(str(classes_dics[j][word])+'\n')
	print "Save model in file : nb/model/nbmodel.txt\n"
	model.close()







