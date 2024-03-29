import numpy as np
import tensorflow as tf
import data_helper
import tf_metrics


class TextCNN(object):
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
		# placeholders for input, output and dropout
		self.input_x = tf.placeholder(
			tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(
			tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		# L2正则损失(可选)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			# load embedding
			embedding_matrix = data_helper.load_embedding(
				embedding_file='./embedding/sgns.wiki.char',
				word_index_path='./data/vocab.bin',
				max_features=10e6, max_document_length=200
			)
			# W = tf.Variable(embedding_matrix, name='W')
			# W = tf.cast(W, 'float32')
			W = tf.Variable(embedding_matrix, dtype=tf.float32, name='W')
			# embedded_chars = tf.nn.embedding_lookup(W,x[:37])
			# (37,200,300,1)
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		# filter_sizes = [3,4,5]
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):
				# Convolution Layer
				# embedding_size = 300
				# num_filters = 32
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				conv = tf.nn.conv2d(
					# input=self.embedded_chars_expanded,
					input=self.embedded_chars_expanded,
					filter = W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='conv')
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		with tf.name_scope('output'):
			W = tf.get_variable(
				'W',
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		# Calculate mean cross-entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
		
		# with tf.name_scope('more_metric'):
		# 	self.mpa, self.pa = tf.metrics.mean_per_class_accuracy(self.predictions, tf.argmax(self.input_y, 1),num_classes)
		# 	*_, self.f = tf_prf.prf(tf.argmax(self.input_y, 1), self.predictions, num_classes, pos_indices=range(num_classes))
		# _, self.f = tf_metrics.f1(tf.argmax(self.input_y, 1), self.predictions, num_classes, pos_indices=range(num_classes))

		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')



