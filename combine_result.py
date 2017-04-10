import math
import pandas as pd
from pandas import DataFrame
from pandas import Series
import csv
import codecs
import os
import numpy as np 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

pos_raw = list(codecs.open("Data/C000013_dev.txt", "r", "utf-8").readlines())
pos_label = np.ones((1,len(pos_raw)),dtype=np.int)
neg_raw = list(codecs.open("Data/C000024_dev.txt", "r", "utf-8").readlines())
neg_label = np.zeros((1,len(neg_raw)),dtype=np.int)
raw_data = np.append(pos_raw, neg_raw)
label = np.append(pos_label[0], neg_label[0])

padding = max(len("Class"), len(str(1)))
label_pad = ['{0:<{1}}'.format("1.0", padding)]
for l in label:
	label_pad.append(str(l))


raw_data = np.append(["F1 Score"], raw_data)
raw_data = np.append(["Raw Data"], raw_data)
label = np.append(["1.0"], label)
label = np.append(["Class"], label)

nb_result = []
knn_result = []
cnn_result = []
svm_result = []
result_num = 0
col = []
F1s = []
if os.path.exists("./result/nb_result.txt"):
	nb_result = list(open("./result/nb_result.txt", "r").readlines())
	nb_result = np.append(['Naive Bayes'], nb_result)
	result_num += 1
	col.append("Naive Bayes")
	F1s.append(nb_result[1].strip())
if os.path.exists("./result/knn_result_cosin.txt"): 
	knn_result = list(open("./result/knn_result_cosin.txt", "r").readlines())
	knn_result = np.append(['KNN'],knn_result)
	result_num += 1
	col.append("KNN")
	F1s.append(knn_result[1].strip())
if os.path.exists("./result/svm_result.txt"):
	svm_result = list(open("./result/svm_result.txt", "r").readlines())
	svm_result = np.append(['SVM'],svm_result)
	result_num += 1
	col.append("SVM")
	F1s.append(svm_result[1].strip())
if os.path.exists("./result/cnn_result.txt"):
	cnn_result = list(open("./result/cnn_result.txt", "r").readlines())
	cnn_result = np.append(['CNN'],cnn_result)
	result_num += 1
	col.append("CNN")
	F1s.append(cnn_result[1].strip())
columns = np.column_stack((raw_data, label))
# print len(columns)
# print len(nb_result)
# print len(knn_result)
# print len(svm_result)
# print len(cnn_result)
if len(nb_result) != 0:
	columns = np.column_stack((columns, nb_result))
if len(knn_result) != 0:
	columns = np.column_stack((columns, knn_result))
if len(svm_result) !=0:
	columns = np.column_stack((columns, svm_result))
if len(cnn_result) != 0:
	columns = np.column_stack((columns, cnn_result))
if result_num != 0:
	# s = pd.Series(F1s)
	# print s
	print ""
	#pd.set_option('colheader_justify', 'left')
	data = DataFrame([F1s],index=['F1 Score'],columns=col)
	print data
	print ""
out_path = "result/all_result.csv"
with open(out_path, 'w') as f:
    f.write(codecs.BOM_UTF8)
    #csv.writer(f).writerow(["Raw Data".encode('utf-8'), "Class".encode('utf-8'),"Naive Bayes","KNN","CNN"])
    csv.writer(f).writerows(columns)

