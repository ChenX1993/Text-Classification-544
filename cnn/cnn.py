# -- coding: utf-8 --

import time
import os
import datetime
import my_data_helpers
import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import learn
from my_text_cnn import CNN_obj

#------------------- Parameters ------------------------------

# Parameters for data loading
tf.flags.DEFINE_string("class_1_file", "Data/C000013_train.txt", "source for class 1")
tf.flags.DEFINE_string("class_2_file", "Data/C000024_train.txt", "source for class 2")
tf.flags.DEFINE_float("dev_percentage", 0.1, "Percentage of the training data to use for validation")


#Parameters for Model Hyperparameters
tf.flags.DEFINE_integer("embed_dimension", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_size", "3,4,5", "Filter_size, separated by comma")
tf.flags.DEFINE_integer("num_filter", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

#Parameters for training
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_integer("num_epoch", 1, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_point", 15, "Evaluate model on dev set after these steps")
tf.flags.DEFINE_integer("checkpoint", 15, "Save model after these steps")
tf.flags.DEFINE_integer("num_checkpoint", 5, "Number of saved checkpoint")

#Parameters for Misc
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS  #所有parameter
FLAGS._parse_flags()
print("\n\nParameters:")
for attribute, value in sorted(FLAGS.__flags.items()):
	print("{} = {}".format(attribute.upper(), value))
print('\n')

#------------------- Load data and preprocess -------------------------

#Load
print ("Loading data ...")
x_text, y_label = my_data_helpers.load_data(FLAGS.class_1_file, FLAGS.class_2_file)
"""
build vocabulary
whole_text: list: 一个sample是一个element
"""
#最长sample 长度
max_sample_length = 0
for text in x_text:
	word_list = text.split(" ")
	max_sample_length = max(max_sample_length, len(word_list))

# print max_sample_length
#high-level machine learning API (tf.contrib.learn)
#定义一个将所有文档映射成词索引的序列的实例，该序列最大为 max_sample_length
#使用分词器对文档进行分词，提取词袋，统计词被应用的次数,提取每一项文档的前 max_sample_length 个词的索引组合成新的X_train词袋的表现形式为：{"Abbott":1, "of":2, "Farnham":3, ....}
#定义分词器
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sample_length)
#进行分词， 映射 返回映射结果
x_map = np.array(list(vocab_processor.fit_transform(x_text)))
#fit_transform可以提取词袋

#Random shuffle
np.random.seed(10) #每次产生的随机数相同
#重新安排smaple的顺序
shuffle_key = np.random.permutation(np.arange(len(y_label)))
x_shuffle = x_map[shuffle_key]
y_shuffle = y_label[shuffle_key]
#print x_shuffle

#把输入分成 train 和 dev 两部分
index = int(FLAGS.dev_percentage * float(len(y_label))) * -1
x_train = x_shuffle[:index]
x_dev = x_shuffle[index:]
y_train = y_shuffle[:index]
y_dev = y_shuffle[index:]
print ("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print ("Train Size: {:d}".format(len(y_train))) +"   "+("Develop Size: {:d}".format(len(y_dev)))

#------------------------ Train --------------------------------




"""
session 是 graph 的执行环境， 包含变量和队列状态
一个session执行一个graph

graph 包含操作和张量
"""
"""
allow_soft_placement: 允许 TensorFlow 回退到特定操作的设备，如果在优先设备不存在时。e.g. 用于GPU，如果没有，接受CPU
log_device_placement: 记录运行的设备（CPU or GPU）
"""

with tf.Graph().as_default():
	session_config = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement,
									log_device_placement = FLAGS.log_device_placement)
	session = tf.Session(config = session_config)
	# sequence_length : sample 长度
	filter_size_list = list(map(int, FLAGS.filter_size.split(",")))
	size_list = FLAGS.filter_size.split(",")
	filter_size_list = []
	for size in size_list:
		filter_size_list.append(int(size))
	with session.as_default():
		# 实例化
		cnn = CNN_obj(sequence_length = x_train.shape[1], num_class = y_train.shape[1],
					  vocab_size = len(vocab_processor.vocabulary_), embedding_size = FLAGS.embed_dimension,
					  filter_size = filter_size_list, num_filter = FLAGS.num_filter, 
					  l2_reg_lambda = FLAGS.l2_reg_lambda)
		
		#定义如何去最优化我们网络的损失函数,使用Adam优化器 进行梯度计算
		global_step = tf.Variable(0, name = "global_step", trainable = False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradient = optimizer.compute_gradients(cnn.loss)
		#train_optimizer 是一个训练步骤。TensorFlow 会自动计算出哪些变量是“可训练”的，并计算它们的梯度。
		#通过定义 global_step 变量并将它传递给优化器，我们允许TensorFlow处理我们的训练步骤。我们每次执行 train_optimizer 操作时，global_step 都会自动递增1。
		train_optimizer = optimizer.apply_gradients(gradient, global_step = global_step)

		# 汇总输出 跟踪在各个训练和评估阶段，损失值和正确值是如何变化的 
		# SummaryWriter 函数来将它们写入磁盘
		#time = str(int(time.time()))
		#output_dir = "./runs/" + time
		output_dir = os.path.abspath(os.path.join(os.path.curdir, "model"))
		print "Save Output to {}\n".format(output_dir)



		# 跟踪梯度值变化
		gradient_summary = []
		for grad, value in gradient:
			if grad is not None:
				grad_histogram = tf.summary.histogram("{}/grad/histogram".format(value.name), grad)
				sparsity = tf.summary.scalar("{}/grad/sparsity".format(value.name), tf.nn.zero_fraction(grad))
				gradient_summary.append(grad_histogram)
				gradient_summary.append(sparsity)
		gradient_merge = tf.summary.merge(gradient_summary)
		# loss and accuracy
		#Outputs a Summary protocol buffer containing a single scalar value.
		# summary 叠加次数
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		accuracy_summary = tf.summary.scalar("accuracy", cnn.accuracy)

		# train save
		train_summary_data = tf.summary.merge([loss_summary, accuracy_summary, gradient_merge])
		train_summary_dir = os.path.join(output_dir, "summaries", "train")
		train_summary_write = tf.summary.FileWriter(train_summary_dir, session.graph)
		#develop save
		dev_summary_data = tf.summary.merge([loss_summary, accuracy_summary])
		# dev_summary_dir = os.path.join(output_dir, "summaries", "dev")
		# dev_summary_write = tf.summary.FileWriter(dev_summary_dir, session.graph)

		#checkpint dir # tensorflow 默认路径存在，所以先创建
		#保存模型的参数以备以后恢复。检查点可用于在以后的继续训练，或者提前来终止训练
		checkpoint_dir = os.path.abspath(os.path.join(output_dir, "checkpoints"))
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		prefix = os.path.join(checkpoint_dir, "model")
		checkpoint_save = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoint)

		# save vocabulary
		vocab_processor.save(os.path.join(output_dir,"vocab"))

		#初始化变量
		session.run(tf.global_variables_initializer())


		#定义一个训练函数，用于单个训练步骤，在一批数据上进行评估，并且更新模型参数。
		def single_train_step(x_input, y_input):
			#feed_dic 包含了需要传入到网络中的数据。必须为所有的占位符节点提供值
			feed_dic = {cnn.x_input : x_input, cnn.y_input : y_input, cnn.dropout_keep_prob : FLAGS.dropout_keep_prob}

			#Runs operations and evaluates tensors in fetches.
			#This method runs one "step" of TensorFlow computation
			#by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches
			#substituting the values in feed_dict for the corresponding input values.
			#优化器，
			#session.run 来执行 优化器
			_, step, summary, loss, accuracy = session.run(
				[train_optimizer, global_step, train_summary_data, cnn.loss, cnn.accuracy], feed_dic)
			curr_time = datetime.datetime.now().isoformat()
			print ("step # {} : loss {:g}, accuracy {:g}".format(step, loss, accuracy))
			train_summary_write.add_summary(summary, step)

		# evaluate model
		#评估任意数据集的损失值和真确率，在交叉验证数据集上验证。
		def single_dev_step(x_input, y_input, writer = None):
			#禁用dropout
			feed_dic = {cnn.x_input : x_input, cnn.y_input : y_input, cnn.dropout_keep_prob : 1.0}
			step, summary, loss, accuracy = session.run([global_step, dev_summary_data, cnn.loss, cnn.accuracy], feed_dic)
			print ("step # {} : loss {:g}, accuracy {:g}".format(step, loss, accuracy))
			# if writer:
			# 	writer.add_summary(summary, step)

		#generate batches
		batches = my_data_helpers.batch_iteration(zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epoch)

		#Training interation. for each batch
		"""
		完整的训练过程。对数据集进行批次迭代操作，为每个批处理调用一次 train_step 函数
		每 evaluate_point（100） 次去评估一下训练模型。
		"""
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			single_train_step(x_batch, y_batch)
			curr_step = tf.train.global_step(session, global_step)
			if curr_step % FLAGS.evaluate_point == 0:
				print('\nEvaluation:')
				single_dev_step(x_dev, y_dev)
				print('\n')
			if curr_step % FLAGS.checkpoint == 0:
				save_path = checkpoint_save.save(session, prefix, global_step = curr_step)
				print ("Save model checkpoint to {}\n".format(save_path))








