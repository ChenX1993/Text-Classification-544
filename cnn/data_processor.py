# -- coding: utf-8 --
import numpy as np 
import re
import itertools
from collections import Counter
import codecs

"""
load data
splits into words
return split sentences and label
输入处理完的分词数据
"""
def load_data(class_1_file, class_2_file):
	#导入样本
	class_1_samples = list(codecs.open(class_1_file, "r", "utf-8").readlines())
	class_2_samples = list(codecs.open(class_2_file, "r", "utf-8").readlines())
	# 去换行符
	#class_1_samples = [s.strip() for s in class_1_samples]
	class_1_samples = map(lambda s : s.strip(), class_1_samples)
	class_2_samples = map(lambda s : s.strip(), class_2_samples)
	# for line in class_2_samples:
	# 	line = line.strip()
	#class_1_samples = map(str.strip, class_1_samples)
	whole_text = class_1_samples + class_2_samples
	# 所有class 1 label 是【0，1】
	# 所有class 2 label 是【1，0】
	class_1_labels = []
	for _ in class_1_samples:
		class_1_labels.append([0,1])
	class_2_labels = []
	for _ in class_2_samples:
		class_2_labels.append([1,0])
	#label 数组拼接
	y = np.concatenate([class_1_labels, class_2_labels], 0)
	#print y
	#class_1_labels = [[0,1] for _ in class_1_samples]
	#print class_1_labels
	return [whole_text, y]
#load_data("data/rt-polaritydata/C000013_pre_short.txt", "data/rt-polaritydata/C000024_pre_short.txt")
#rt-polarity.neg



"""
shuffle_batch : 降低各批次输入样本之间的相关性(如果训练数据之间相关性很大，可能会让结果很差、泛化能力得不到训练、这时通常需要将训练数据打散，称之为shuffle_batch)。
构建词汇索引表，将每个单词映射到 0 ~ 18765 之间（18765是词汇量大小），那么每个句子就变成了一个整数的向量。
"""
def batch_iteration(input_data, batch_size, epochs_num):
	input_data = np.array(input_data)
	size = len(input_data)
	batches_per_epoch = int((len(input_data) - 1)/batch_size) + 1
	for each_epoch in range(0, epochs_num):
		#打乱顺序
		shuffle = np.random.permutation(np.arange(size))
		shuffle_data = input_data[shuffle]

		for batch_count in range(0, batches_per_epoch):
			s_index = batch_count * batch_size
			e_index = min(size, (batch_count + 1) * batch_size)
			#每个iteration 抛回结果
			yield shuffle_data[s_index : e_index]

def batch_iteration_for_eval(input_data, batch_size):
	input_data = np.array(input_data)
	size = len(input_data)
	batch_num = int((len(input_data) - 1)/batch_size) + 1
	for batch_count in range(0, batch_num):
		s_index = batch_count * batch_size
		e_index = min(size, (batch_count + 1) * batch_size)
		yield input_data[s_index : e_index]








