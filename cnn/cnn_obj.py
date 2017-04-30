# -- coding: utf-8 --
import tensorflow as tf 
import numpy as np 

class CNN_obj(object):
	def __init__(self, sequence_length, num_class, vocab_size, embedding_size,
				filter_size, num_filter, l2_reg_lambda):

		self.x_input = tf.placeholder(tf.int32, [None, sequence_length],name = "x_input")
		self.y_input = tf.placeholder(tf.float32, [None, num_class], name = "y_input")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

		with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
			self.W_vec = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name = "W_vec")
			self.embed_char = tf.nn.embedding_lookup(self.W_vec, self.x_input)
			self.embed_char_expand = tf.expand_dims(self.embed_char, -1)

		pool_output = []
		
		for i, size in enumerate(filter_size):
			with tf.name_scope("conv-maxpool-layer%s" % size):  #为每个size的核创建名称范围
				#Convolution Layer
				filter_shape = [size, embedding_size, 1, num_filter] 
				filter_core = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1),name = "filter_core") 
				bias = tf.Variable(tf.constant(0.1, shape=[num_filter]),name = "bias")
				conv = tf.nn.conv2d(self.embed_char_expand, filter_core, strides = [1, 1, 1, 1], padding = "VALID", name = "conv")
				#nonlinearity
				non_linear = tf.nn.relu(tf.nn.bias_add(conv, bias), name = "relu")
				#输出维度：[1, sequence_length - filter_size + 1, 1, 1]
				#Maxpooling over the outputs 最大池化处理
				#张量维度：[batch_size, 1, 1, num_filters] -- 特征向量，最后一个维度就是特征
				maxpool = tf.nn.max_pool(non_linear, ksize = [1, sequence_length - size + 1, 1, 1],
						  strides = [1, 1, 1, 1], padding = "VALID", name = "maxpool")
				pool_output.append(maxpool)

		# Combine all features 合并各个卷积核所得到的特征向量
		total_filters = num_filter * len(filter_size)
		# 合并list
		self.h_pool = tf.concat(pool_output, 3)
		#重新定义格式：
		self.flat_h_pool = tf.reshape(self.h_pool, [-1, total_filters])
	
		with tf.name_scope("dropout_layer"):
			self.dropout = tf.nn.dropout(self.flat_h_pool, self.dropout_keep_prob)

		l2_loss = tf.constant(0.0)
		with tf.name_scope("output"):
			#Gets an existing variable with these parameters or create a new one as initializer
			weight = tf.get_variable("weight", shape = [total_filters, num_class], 
									 initializer = tf.contrib.layers.xavier_initializer())
			l2_loss += tf.nn.l2_loss(weight)
			bias = tf.Variable(tf.constant(0.1, shape = [num_class]), name = "bias")
			l2_loss += tf.nn.l2_loss(bias)
			#f.nn.xw_plus_b是一个很方便的函数，实现 Wx + b 操作
			self.score = tf.nn.xw_plus_b(self.dropout, weight, bias, name = "score")
			self.predict = tf.argmax(self.score, 1, name = "predict")
		#loss and accuracy
		with tf.name_scope("loss"):
			#计算每个类别的交叉损失熵,输入给定的分数和输入的正确标签
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.score, labels = self.y_input)
			#计算损失值的平均值
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
		#定义正确率函数 训练阶段和测试阶段来跟踪模型的性能
		with tf.name_scope("accuracy"):
			correct_predic = tf.equal(self.predict, tf.argmax(self.y_input, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predic, "float"), name = "accuracy")








	