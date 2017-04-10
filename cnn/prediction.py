# -- coding: utf-8 --
import os
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf 
from tensorflow.contrib import learn
import codecs
import csv
import sys
import data_processor
reload(sys)
sys.setdefaultencoding('utf-8')


#---------------------- Parameters -------------------------------

# Data
# Misc
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Evaluation
tf.flags.DEFINE_string("checkpoint_dir", "cnn/model/checkpoints/", "checkpoint directory")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")

# checkpoint_dir = sys.argv[1]
tf.flags.DEFINE_string("class_1_file", "Data/C000013_dev.txt", "Data source for class 1")
tf.flags.DEFINE_string("class_2_file", "Data/C000024_dev.txt", "Data source for class 2")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# print("\nParamenters:")
# for attribute, value in sorted(FLAGS.__flags.items()):
# 	print("{} = {}".format(attribute.upper(), value))
# print('\n')

#Load data
x_dev, y_real = data_processor.load_data(FLAGS.class_1_file, FLAGS.class_2_file)
#[[0,1][0,1][1,0][1,0]] ==> [1,1,1,1,0,0,0,0]
y_real = np.argmax(y_real, axis = 1)
# print y_real
vocab_path = FLAGS.checkpoint_dir + "../vocab"
#加载与训练一样的分词器
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#分词与映射
x_map = np.array(list(vocab_processor.transform(x_dev)))

print ("\nEvaluating ... \n")

#----------------------- Evaluation --------------------------------
#最新的model
checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# print "Latest checkpoint file is : " + FLAGS.checkpoint_dir
# print "Latest model is : " + checkpoint

graph = tf.Graph()
with graph.as_default():
	session_config = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement, 
									log_device_placement = FLAGS.log_device_placement)
	session = tf.Session(config = session_config)
	with session.as_default():
		#加载meta data 和变量  Recreates a Graph saved in a MetaGraphDef proto
		#设置当前graph为训练得到的cnn模型
		meta_graph = tf.train.import_meta_graph("{}.meta".format(checkpoint))
		meta_graph.restore(session, checkpoint)
		#预测结果
		predict = graph.get_operation_by_name("output/predict").outputs[0]

		#generate batch
		batches = data_processor.batch_iteration_for_eval(list(x_map), FLAGS.batch_size)

		#获取占位符
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		x_input = graph.get_operation_by_name("x_input").outputs[0]
		
		all_predict = []
		for x_batch in batches:
			batch_predic = session.run(predict, {x_input : x_batch, dropout_keep_prob : 1.0})
			all_predict = np.concatenate([all_predict, batch_predic])
			#all_predict.append(batch_predic)
			#print all_predict
#计算准确度
# correct_predict = float(sum(all_predict == y_real))
# print correct_predict

correct_predict = 0
for i in range(0, len(y_real)):
	if (all_predict[i] == y_real[i]):
		correct_predict += 1
print ""
print "Predict Results:"		
print("Total number of test cases : {}".format(len(y_real)))
print("Total number of correct predictions : {}".format(correct_predict))
acc = float(correct_predict)/float(len(y_real))


TP = 0
FP = 0
FN = 0
for i in range(len(y_real)):
	if (all_predict[i] == 1):
		if y_real[i] == 1:
			TP += 1
		else:
			FP += 1
	elif y_real[i] == 1:
		FN += 1
P = float(TP) / (TP + FP)
R = float(TP) / (TP + FN)
F1 = 2.0 * P * R / (P + R)
#print("Accuracy : {:g}".format(acc))
print("F1 Score : {:g}".format(round(F1,6)))

#保存到csv
# saved_predict = np.column_stack((np.array(x_dev), all_predict))
# #output_file = os.path.join(FLAGS.checkpoint_dir,"..","prediction.csv")
# output_file = "./data/predictions/cnn_prediction.csv"
# with open(output_file,'w') as out:
# 	out.write(codecs.BOM_UTF8)
# 	csv.writer(out).writerows(saved_predict)

# print len(all_predict)
#save to txt
with open("result/cnn_result.txt",'w') as output:
	output.write(str(round(F1,6)) +'\n')
	for x in all_predict:
		output.write(str(int(x))+'\n')






