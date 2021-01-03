#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet POs-Tagginf with an averaged perceptron
@authors: GALLIENNE Romane, GUITEL Cécile
"""


# corpus:
# https://universaldependencies.org/treebanks/fr_gsd/index.html#ud-french-gs
# https://github.com/UniversalDependencies/UD_French-GSD

# bibliography: 
# https://moodle.u-paris.fr/pluginfile.php/649525/mod_resource/content/1/perceptron.pdf (training pseudo-code algorithm)
# https://www.nltk.org/_modules/nltk/tag/perceptron.html
# https://www.guru99.com/pos-tagging-chunking-nltk.html
# https://becominghuman.ai/part-of-speech-tagging-tutorial-with-the-keras-deep-learning-library-d7f93fa05537 (list of features)

import random
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def get_word_vector(sentence, word, index):
	"""Calculates the features of a given word, returns that vector.
	Features: word, word before, word after, Capitalized, UPPER, pre- and 
	suffixes with len from 1 to 3 (if they exist), len = 1 or 2 or 3 or more.

	sentence: list of Strings (words), the original sentence.
	word: String, the word.
	index: index of word in the sentence. Taken from the corpus, so originally, 
	index of sentence[0] is 1. Hence the line: index -= 1
	"""

	vector = {}
	len_word = len(word)
	len_sentence = len(sentence)
	index -= 1

	vector["word="+word] = 1

	if index == 0:
		vector["is_first"] = 1
	else:
		vector["word-1="+sentence[index-1]] = 1

	if index == len_sentence -1:
		vector["is_last"] = 1
	else:
		vector["word+1="+sentence[index+1]] = 1

	if word[0].isupper():
		vector["is_capitalized"] = 1
        
	if word.isupper():
        	vector["is_uppercase"] = 1

	if len_word == 1:
		vector["len=1"] = 1
	elif len_word == 2:
		vector["len=2"] = 1
	elif len_word == 3:
		vector["len=3"] = 1
	else:
		vector["len>3"] = 1
	

	if len_word > 0:
		vector["prefix1="+word[0:1]] = 1

	if len_word > 1:
		vector["prefix2="+word[0:2]] = 1

	if len_word > 2:
		vector["prefix3="+word[0:3]] = 1

	if len_word > 0:
		vector["suffix1="+word[-1::]] = 1

	if len_word > 1:
		vector["suffix2="+word[-2::]] = 1

	if len_word > 2:
		vector["suffix3="+word[-3::]] = 1

	#print(vector)

	return vector



def predict_tag(vector, weights, tag_list):
	"""Predicts and returns the tag of a word.

	vector: features of the word, as calculated/formatted by get_word_vector
	weights: the weights for each feature in the prediction
	tag_list: possible tags
	"""

	scores = {}
	for tag in tag_list:
		scores[tag] = 0
		for feature in vector:
			if feature not in weights:
				weights[feature] = {}
			weights_feature = weights[feature]
			scores[tag] += weights_feature.get(tag,0) * vector[feature]

	return max(scores, key=lambda tag: (scores[tag], tag))



def add_vector_to_weights(vector, weights, tag, factor):
	"""Adds or substract (for factor = 1 and -1) a vector to a set of weights. 
	Auxiliary fonction for train.

	vector: word vector, as calculated/formatted by get_word_vector()
	weights: the weights for each feature in the prediction
	tag: the tag that needs to be reevaluated
	factor: which way the tag needs to be reevaluated, here, 1 or -1
	"""

	for feature in vector:
		if feature not in weights:
			weights[feature] = {}
		weights_feature = weights[feature]
		weights_feature[tag] = weights_feature.get(tag, 0) + vector[feature]*factor



def add_weights_to_average(average, weights):
	"""Adds the calculated weights to the average, to smooth
	variations out and allow for reaching a limit. Auxiliary
	fonction of train.

	average: the current average
	weights: the calculated weights to be added
	"""

	for feature in weights:
		weights_feature = weights[feature]
		for tag in weights[feature]:
			if feature not in average:
				average[feature] = {}
			average_feature = average[feature]
			average_feature[tag] = average_feature.get(tag, 0) + weights_feature[tag]



def train(train_vectors, tag_list, MAX_EPOCH = 1, evaluate_epochs = False, dev_vectors = None):
	"""Creates and return the weights to score each tags and predict the best 
	one. Weights are averaged by adding each temporary value of them to each
	other. If evaluate_epochs = True, evaluates at every epoch to find the best
	MAX_EPOCH (up to the value given as parameter of the function).

	train_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data, to get the weights
	tag_list: list of existing tags
	MAX_EPOCH: super parameter, number of times the algorithm goes 
	through the whole corpus
	evaluate_epochs: whether to evaluate the weights at each epoch
	dev_vectors: same, but a separate set, to evaluate the accuracy of the 
	weights depending on the number of epochs they were trained on
	"""

	average = {}
	if evaluate_epochs:
		results = {}
		print("For MAX_EPOCH in "+str(range(0,MAX_EPOCH))+"\nn_epochs\taccuracy")

	for epoch in range(0, MAX_EPOCH):
		#print("epoch: "+str(epoch))
		weights = {}
		random.shuffle(train_vectors)
		index_word = 0
		n_words = len(train_vectors)

		for word in train_vectors:
			#print("epoch "+str(epoch)+"/"+str(MAX_EPOCH)+": "+str(index_word)+"/"+str(len(train_vectors))+" words")
			index_word += 1
			predicted_tag = predict_tag(word[0], weights, tag_list)
			gold_tag = word[1]
			if predicted_tag != gold_tag:
				add_vector_to_weights(word[0], weights, predicted_tag, -1)
				add_vector_to_weights(word[0], weights, gold_tag, +1)
		#RAF la ligne suivante est dans la boucle for dans le pseudo-code du prof, ??
		add_weights_to_average(average, weights)

		if evaluate_epochs:
			results[epoch] = evaluate(average, dev_vectors, tag_list)
			print(str(epoch)+"\t\t"+str(results[epoch]))

	if evaluate_epochs:
		print("best MAX_EPOCH: "+str(max(results, key=lambda n_epochs: (results[epoch], epoch))))

	return average



def get_data_from_file(file = "./fr_gsd-ud-train.conllu"):
	"""Extracts and returns the data from a conllu file. Formats them in 
	a list of lists (sentences) of dictionnaries (words).
	Example:
	[[{index: 1, word: sentence1word1, gold_POS : ADV}, 
   {index: 2, word : sentence1word2, gold_POS : DET}, ...],
	 [{index:1; word: sentence2word1, ...}, ...],
	 ...]

	file: the file path
	"""

	data = []

	with open(file, "r") as raw_file:
		raw_content = raw_file.read()
		sentences = raw_content.split("\n\n")
		for sentence in sentences:
			sentence_data = []
			for line in sentence.split("\n"):
				if line != "" and line[0] != "#":
					tabs = line.split("\t")
					if tabs[3] != "_":
						to_append = {}
						to_append["index"] = int(tabs[0])
						to_append["word"] = tabs[1]
						to_append["gold_POS"] = tabs[3]
						sentence_data.append(to_append)
			data.append(sentence_data)
	
	return data



def get_vectors_from_data(data):
	"""Creates and returns the word vectors from extracted data, in the form of 
	a list of tuples (word_vector, gold_POS)

	data: semi-raw data, as extracted/formatted by get_data_from_file
	"""	

	vectors = []
	for sentence_data in data:
		sentence = []
		for word_data in sentence_data:
			sentence.append(word_data["word"])
		for word_data in sentence_data:
			word_vector = get_word_vector(sentence, word_data["word"], word_data["index"])
			to_append = (word_vector, word_data["gold_POS"])
			vectors.append(to_append)
			#print(vectors)
	return vectors



def evaluate(weights, test_vectors, tag_list, confusion_matrix = False):
	"""Calculates the precision of the calculated weights on a testing corpus 
	by counting the number of bad answers. If confusion_matrix = True,
	calculates and save that error matrix.

	weights: the weights to evaluate
	test_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data
	tag_list: list of existing tags
	"""

	good = 0
	random.shuffle(test_vectors)

	#if confusion_matrix:
		#pos_pred_gold is a list of tuple (pred_pos,gold_pos) that is used to
		#create a confusion matrix to see performance of tagging
		#pos_pred_gold = []

	for word in test_vectors:
		predicted_tag = predict_tag(word[0], weights, tag_list)
		gold_tag = word[1]
		#if confusion_matrix:
			#pos_pred_gold.append((predicted_tag,gold_tag))
		if predicted_tag == gold_tag:
			good += 1

	#if confusion_matrix:	
		#state_of_tagging = matrix_confusion(pos_pred_gold, tag_list) #permet de visualiser les erreurs d'étiquetage

	print("Good answers: "+str(good)+"/"+str(len(test_vectors)))

	return good



def matrix_confusion(decision_corpus, tag_list):
	"""Calculates and returns a confusion matrix to analyse tagging performance.

	decision_corpus: list of tagging decisions, saved as tuples (pred_pos, gold
	pos)
	tag_list: list of existing tags
	"""

	matrix = np.zeros((len(tag_list), len(tag_list)))
	
	gold_pos = []
	pred_pos = []
	
	for pair in decision_corpus:
 		pred_pos.append(pair[0])
 		gold_pos.append(pair[1])
	
	matrix += confusion_matrix(gold_pos, pred_pos, labels = tag_list)
	matrix_plot(matrix, tag_list, "Confusion matrix of PoSTagging")

	return matrix



def matrix_plot(confusion_matrix, tag_list, graph_title):
	"""Prints and saves on desktop a heatmap of the common errors.

	confusion_matrix: confusion matrix, as calculated by matrix_confusion()
	tag_list : list of existing tags
	graph_title : title of the heatmap
	"""
    
	plt.figure(figsize=(12,12))
	plt.xticks(ticks=np.arange(len(tag_list)),labels=tag_list,rotation=90)
	plt.yticks(ticks=np.arange(len(tag_list)),labels=tag_list)
	hm=plt.imshow(confusion_matrix, cmap='Blues', interpolation = None) 
	plt.colorbar(hm) 
	plt.title(graph_title) 
	plt.xlabel("predicted_labels") 
	plt.ylabel("gold_labels") 
	
	for i in range(len(tag_list)): 
		for j in range(len(tag_list)): 
			if confusion_matrix[i, j] > 0:
				text = plt.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="brown") 
	
	plt.savefig(graph_title)



if "__main__" == __name__:
	"""Creates, trains and evaluates an averaged POS-tagging perceptron.
	"""

	start_time = time.time()
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	"""Training"""
	train_data = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors = get_vectors_from_data(train_data)
	#weights = train(train_vectors, tag_list, MAX_EPOCH=50)
	#evaluate(weights, train_vectors, tag_list)

	"""MAX_EPOCH"""
	dev_data = get_data_from_file("./fr_gsd-ud-dev.conllu")
	dev_vectors = get_vectors_from_data(dev_data)
	#weights = train(dev_vectors, tag_list, MAX_EPOCH=5, evaluate_epochs=True, dev_vectors=dev_vectors)
	#evaluate(weights, train_vectors, tag_list)

	"""Evaluation"""
	test_data = get_data_from_file("./fr_gsd-ud-test.conllu")
	test_vectors = get_vectors_from_data(test_data)
	weights = train(test_vectors, tag_list, MAX_EPOCH=50)
	evaluate(weights, test_vectors, tag_list)

	"""Total time evaluation"""
	print("Took "+str(int(time.time()-start_time))+" secondes")

	#To do : validation with MAX_EPOCH and dev_vectors
