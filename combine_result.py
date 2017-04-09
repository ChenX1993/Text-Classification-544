import csv
import codecs
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

nb_result = list(open("./result/nb_result.txt", "r").readlines())
knn_result = list(open("./result/knn_result.txt", "r").readlines())
svm_result = list(open("./result/svm_result.txt", "r").readlines())
cnn_result = list(open("./result/cnn_result.txt", "r").readlines())
raw_data = np.append(["None"], raw_data)
label = np.append(["1.0"], label)
print len(nb_result)
print len(label)
print len(raw_data)
columns = np.column_stack((raw_data, label, nb_result))

out_path = "result/all_result.csv"
with open(out_path, 'w') as f:
    f.write(codecs.BOM_UTF8)
    csv.writer(f).writerow(["Raw Data".encode('utf-8'), "Correct Class".encode('utf-8'),"Naive Bayes","KNN","SVM","CNN"])
    csv.writer(f).writerows(columns)
