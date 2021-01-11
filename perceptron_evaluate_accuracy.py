#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from perceptron_basics import *
from perceptron_serialisation import deserialise_weights

"""This file is dealing with the accuracy evaluation of perceptron"""


def get_decision_corpus(weights, test_vectors, tag_list): #RAF comment/si ajouter in/out vocab ??
	"""Creates and returns a list of decisions taken by the perceptron, in the
	form of dictionaries with the following keys: word_vector, gold_tag,
	predicted_tag.

	weights: the weights to evaluate
	test_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data
	tag_list: list of existing tags
	"""

	decision_corpus = []

	for word in test_vectors:
		decision = {}
		decision["word_vector"] = word[0]
		decision["gold_tag"] = word[1]
		decision["predicted_tag"] = predict_tag(decision["word_vector"], weights, tag_list)
		
		decision_corpus.append(decision)

	return decision_corpus



def get_vocabulary(words):
	"""Creates and return a dictionary of known word vectors. Would have been used
	to differentiate between accuracy on known and unknown words, except going through
	the list each time took too much time.

	words: list of (word_vector, gold_tag) as created/formatted by
	get_vectors_from_data()
	"""

	vocabulary = []
	for word in words:
		vocabulary.append(word[0])

	return vocabulary



def evaluate_accuracy(decision_corpus):
	"""Calculates, prints and returns the number of good guesses. Would ideally discriminate
	between known and unknown vocabulary.

	decision_corpus: list of tagging decisions, saved as dictionaries (word_vec, pred_pos,
	gold_pos), as created/formatted by get_decision_corpus()
	"""

	good_overall = 0
	total_overall = len(decision_corpus)

	for decision in decision_corpus:
		if decision["gold_tag"] == decision["predicted_tag"]:
			good_overall += 1

	print("Accuracy:\t"+str(good_overall)+"/"+str(total_overall))
	return good_overall



def print_errors(decision_corpus):
	"""Prints the incorrect decisions in order to be able to analyze them manually.

	decision_corpus: list of tagging decisions, saved as dictionaries (word_vec, pred_pos,
	gold_pos), as created/formatted by get_decision_corpus()
	"""

	for decision in decision_corpus:
		if decision["gold_tag"] != decision["predicted_tag"]:
			print(decision)



if "__main__" == __name__:
	"""Calculates and prints accuracy results for the serialised perceptron.
	"""

	weights = deserialise_weights()
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	print("Evaluating in-domain on gsd corpus")

	print("On training data/known vocab")
	train_data_gsd = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors_gsd = get_vectors_from_data(train_data_gsd)
	decision_corpus_gsd_train = get_decision_corpus(weights, train_vectors_gsd, tag_list)
	evaluate_accuracy(decision_corpus_gsd_train)

	print("On testing data/new vocab")
	test_data_gsd = get_data_from_file("./fr_gsd-ud-test.conllu")
	test_vectors_gsd = get_vectors_from_data(test_data_gsd)
	decision_corpus_gsd_test = get_decision_corpus(weights, test_vectors_gsd, tag_list)
	evaluate_accuracy(decision_corpus_gsd_test)
	print()
	
	
	print("Evaluating outside-domain on spoken and SRCMF corpus")

	print("Spoken data")
	test_data_spoken = get_data_from_file("./Eval_HorsDomaine/French_Spoken/fr_spoken-ud-test.conllu")
	test_vectors_spoken = get_vectors_from_data(test_data_spoken)
	decision_corpus_spoken = get_decision_corpus(weights, test_vectors_spoken, tag_list)
	evaluate_accuracy(decision_corpus_spoken)
	print_errors(decision_corpus_spoken)

	print("SRCMF data")
	test_data_SRCMF = get_data_from_file("./Eval_HorsDomaine/French_SRCMF/fro_srcmf-ud-test.conllu")
	test_vectors_SRCMF = get_vectors_from_data(test_data_SRCMF)
	decision_corpus_SRCMF = get_decision_corpus(weights, test_vectors_SRCMF, tag_list)
	evaluate_accuracy(decision_corpus_SRCMF)
	print()
