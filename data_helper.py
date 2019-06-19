import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from jieba import cut
from tensorflow.contrib import learn


def clean_str(s):
	"""Clean sentence"""
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	return s.strip().lower()


def load_data_and_labels(filename):
	"""Load sentences and labels"""
	df = pd.read_csv(filename, compression='zip', dtype={'text': object})
	label_count = df['labels'].value_counts()
	use_label = label_count[label_count>0].index
	df = df[df.labels.isin(use_label)].reset_index()

	selected = ['labels', 'text']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drop non selected columns
	df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
	df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe

	# Map the actual labels to one hot labels
	# labels = sorted(list(set(df[selected[0]].tolist())))
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	# x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
	x_raw = df[selected[1]].tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]


def chinese_tokenizer(docs):
	for doc in docs:
		yield list(cut(doc))


# 加载词向量，已经写好 embedding_file
def load_embedding(embedding_file='./embedding/sgns.wiki.char', word_index_path='./data/vocab.bin', max_features=10e6, max_document_length=200):
	def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
	emb_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_file,encoding='utf8') if len(o) > 300)
	embeddings_index = {i: j for i,j in emb_index.items() if len(j)==300 }
	all_embs = np.stack(embeddings_index.values())
	emb_mean,emb_std = all_embs.mean(), all_embs.std()
	embed_size = all_embs.shape[1]

	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, 10, tokenizer_fn=chinese_tokenizer)
	vocab_processor = vocab_processor.restore('data/vocab.bin')
	word_index = vocab_processor.vocabulary_._mapping
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
	for word, i in word_index.items():
		if i >= max_features: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None: embedding_matrix[i] = embedding_vector
	return embedding_matrix


if __name__ == '__main__':
	# input_file = './data/data.csv.zip'
	# load_data_and_labels(input_file)
	# vocab = learn.preprocessing.VocabularyProcessor(10, 0, tokenizer_fn=chinese_tokenizer)
	# x = list(vocab.fit_transform(DOCUMENTS))
	# print(np.array(x))
	pass










