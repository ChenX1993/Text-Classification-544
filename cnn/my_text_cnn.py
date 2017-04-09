# -- coding: utf-8 --
import tensorflow as tf 
import numpy as np 

"""
定义 TextCNN 类，用于超参数配置
定义层：embedding layer, convolutional, max-pooling and softmax layer.
	   嵌入层，卷积层，池化层
"""

"""
sequence_length: 句子长度 （固定）
num_class: 最后一层分类的数目，二分类
vocab_size:词汇量的大小。
			确定词向量嵌入层的大小，最终的总词向量维度是 [vocabulary_size, embedding_size]
embedding_size: 每个词的词向量长度
filter_size: 卷积核每次覆盖几个单词
				对于每个卷积核，有 num_filter 个。
				比如，
				filter_sizes = [3, 4, 5] : 卷积核有三种类型，
				分别是每次覆盖3个单词的卷积核，每次覆盖4个单词的卷积核和每次覆盖5个单词的卷积核。
				卷积核一共的数量是 3 * num_filter 个
num_filter: 每个卷积核的数量
"""
class CNN_obj(object):
	def __init__(self, sequence_length, num_class, vocab_size, embedding_size,
				filter_size, num_filter, l2_reg_lambda = 0.0):
		"""
		定义 占位符
		placeholder for input, output, and dropout
		tf.placeholder() 定义一个占位符变量
		可以使用它向我们的模型输入数据。
		第二个参数是输入张量的形状。None 指该维度的长度可以是任何值。
		在我们的模型中，第一个维度是批处理大小，而使用 None 来表示这个值，说明网络允许处理任意大小的批次。

		在 dropout 层中，使用 dropout_keep_prob 参数来控制神经元的激活程度。
		只在训练的时候开启，在测试的时候禁止它

		Dropout：模型训练时随机让网络某些隐含层节点的权重不工作，
				不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了，
				它是防止模型过拟合的一种常用的trikc。同时对全连接层上的权值参数给予L2正则化的限制。这样做的好处是防止隐藏层单元自适应（或者对称），从而减轻过拟合的程度。
		"""
		self.x_input = tf.placeholder(tf.int32, [None, sequence_length],name = "x_input")
		self.y_input = tf.placeholder(tf.float32, [None, num_class], name = "y_input")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
	
		"""
		embedding layer  初始状态：随机矩阵
		将词汇索引映射到低维度的词向量进行表示
		从数据中学习得到的词汇向量表

		tf.device("/cpu:0") 强制代码在CPU上面执行操作

		tf.name_scope 创建新的名称范围："embedding"，该范围将所有的操作都添加到这个"embedding"节点下面。
		以便在TensorBoard中获得良好的层次结构，有利于可视化。

		W_vec 嵌入矩阵，从数据训练过程中得到的。最开始，使用一个随机均匀分布来进行初始化。
		tf.nn.embedding_lookup 创建实际的嵌入读取操作，这个嵌入操作返回的数据维度是三维张量 [None, sequence_length, embedding_size]

		tf.Variable(init_value, name) 创建图的一个节点
		tf.random_uniform(shape, min, max,type, seed, name) 生成均匀分布随机数

		tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
		在params中查找与ids对应的表示
		partition_strategy 决定了ids分布的方式，如果partition_strategy 是 “mod”,每个 id 按照 p = id % len(params) 

		tf.expand_dims() 为张量+1维: -1 加在后面
		卷积操作 conv2d 需要一个四维的输入数据，维度分别是批处理大小，宽度，高度和通道数。
		在我们嵌入层得到的数据中不包含通道数，所以我们需要手动添加它，所以最终的数据维度是 [None, sequence_length, embedding_size, 1]
		"""
		with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
			self.W_vec = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name = "W_vec")
			self.embed_char = tf.nn.embedding_lookup(self.W_vec, self.x_input)
			self.embed_char_expand = tf.expand_dims(self.embed_char, -1)

		"""
		Create convolutional + maxpool layer for each filter size 
		卷积层池化层：--特征向量
		我们使用的卷积核是不同尺寸的。因为每个卷积核经过卷积操作之后产生的张量是不同维度的，
		所有我们需要为每一个卷积核创建一层网络，最后再把这些卷积之后的觉果合并成一个大的特征向量。

		tf.truncated_normal(shape,mean,stddev,dtype,seed,name)
		截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数 

		tf.constant(), 初始化值为1，<num_filter. (单个核的个数)

		tf.nn.conv2d()
		卷积：局部加权平均
		输入4维数据 和 filter，得到2维卷积值
		输入：[batch, in_height, in_width, in_channels]
		filter: [filter_height, filter_width, in_channels, out_channels]

		non_linear：对卷积结果进行非线性
		tf.nn.bias_add(conv, b)：对输出的二维数据加bias
		tf.nn.relu() 非线性处理。计算整流线性--激活函数
		"""
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
		"""
		Dropout layer 正则化卷积神经网络 
		按照一定的概率来“禁用”一些神经元的发放。可以防止神经元共同适应一个特征，而迫使它们单独学习有用的特征。
		神经元激活的概率，我们从参数 dropout_keep_prob 中得到。训练阶段将其设置为 0.5，在测试阶段将其设置为 1.0（即所有神经元都被激活）。
		"""
		with tf.name_scope("dropout_layer"):
			self.dropout = tf.nn.dropout(self.flat_h_pool, self.dropout_keep_prob)

		""" 
		Final (unnormalized) scores and scores
		使用来自池化层的特征向量（经过Dropout），然后通过全连接层，得到一个分数最高的类别。
		应用softmax函数来将原始分数转换成归一化概率，但这个操作是保护会改变我们的最终预测。
		"""
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

		"""
		计算损失函数和正确率
		"""
		with tf.name_scope("loss"):
			#计算每个类别的交叉损失熵,输入给定的分数和输入的正确标签
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.score, labels = self.y_input)
			#计算损失值的平均值
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
		#定义正确率函数 训练阶段和测试阶段来跟踪模型的性能
		with tf.name_scope("accuracy"):
			correct_predic = tf.equal(self.predict, tf.argmax(self.y_input, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predic, "float"), name = "accuracy")








	