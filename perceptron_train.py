#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from perceptron_serialisation import serialise_weights
from perceptron_basics import *
from perceptron_evaluate_accuracy import *


"""This file deals with training the perceptron. You can change the number of epochs
here, or change the training corpus below in the main function, or opt to test the
evolution of the accuracy based on the number of epochs."""

MAX_EPOCH = 10 #super parameter, number of times the algorithm goes through the whole corpus



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



def train_one_epoch(train_vectors, tag_list):
	"""Calculates and return the weights based on one pass through the corpus.

	train_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data, to get the weights
	tag_list: list of existing tags
	"""

	weights = {}
	random.shuffle(train_vectors)

	for word in train_vectors:
			predicted_tag = predict_tag(word[0], weights, tag_list)
			gold_tag = word[1]
			if predicted_tag != gold_tag:
				add_vector_to_weights(word[0], weights, predicted_tag, -1)
				add_vector_to_weights(word[0], weights, gold_tag, +1)

	return weights



def train(train_vectors, tag_list, max_epoch=MAX_EPOCH, evaluate_epochs=False, dev_vectors=None):
	"""Calculates and return the weights that will allow to score each tags and
	predict the best one. Weights are averaged over several epochs.

	If evaluate_epochs = True, evaluates at every epoch to find the best
	MAX_EPOCH (up to the value given as parameter of the function).

	train_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data, to get the weights
	tag_list: list of existing tags
	evaluate_epochs: whether to evaluate the weights at each epoch
	dev_vectors: same as train_vectors, but a separate set, to evaluate the accuracy
	of the weights depending on the number of epochs they were trained on
	"""

	average = {}
	if evaluate_epochs:
		accuracy_by_epoch = {}
		print("For MAX_EPOCH in "+str(range(0,max_epoch))+"\nn_epochs\taccuracy")

	for epoch in range(0, max_epoch):
		#print("epoch: "+str(epoch+1))
		weights = train_one_epoch(train_vectors, tag_list)
		
		"""In the pseudo code in the lesson, the following line is placed inside the loop above.
		As it dramatically increased training time, we left it in the outer loop.
		As such, it averages between each epochs instead of each word."""
		add_weights_to_average(average, weights)

		if evaluate_epochs:
			accuracy_by_epoch[epoch] = evaluate_accuracy(get_decision_corpus(average, dev_vectors, tag_list))
			print(str(epoch)+"\t\t" + str(accuracy_by_epoch[epoch]))

	if evaluate_epochs:
		print("best MAX_EPOCH: " + str(max(results, key=lambda n_epochs: (accuracy_by_epoch[epoch], epoch))))

	return average



if "__main__" == __name__:
	"""Creates, trains and saves an averaged POS-tagging perceptron.
	"""
	
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	#Edit the file paths to change the corpus:
	train_data = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors = get_vectors_from_data(train_data)

	#To evaluate the evolution of the accuracy as the epochs go:
	"""dev_data = get_data_from_file("./fr_gsd-ud-dev.conllu")
	dev_vectors = get_vectors_from_data(dev_data)
	weights = train(train_vectors, tag_list, max_epoch=50, evaluate_epochs=True, dev_vectors=dev_vectors)
	"""

	#To calculate new weights:
	weights = train(train_vectors, tag_list)
	serialise_weights(weights)
