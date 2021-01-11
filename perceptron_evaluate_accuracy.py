#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from perceptron_basics import *
from perceptron_serialisation import deserialise_weights

"""This program evaluates the accuracy of the perceptron on several corpus. Edit the file paths
in the main function to switch domains or change the corpus."""


def evaluate_accuracy(decision_corpus):
	"""Calculates, prints (depending) and returns the number of good guesses. Would ideally
	discriminate between known and unknown vocabulary.

	decision_corpus: list of tagging decisions, saved as dictionaries (word_vec, pred_pos,
	gold_pos), as created/formatted by get_decision_corpus()
	"""

	good_overall = 0
	total_overall = len(decision_corpus)

	for decision in decision_corpus:
		if decision["gold_tag"] == decision["predicted_tag"]:
			good_overall += 1

	print("Accuracy:\t" + str(good_overall) + "/" + str(total_overall))
	
	return good_overall



if "__main__" == __name__:
	"""Calculates and prints accuracy results for the serialised perceptron.
	"""

	weights = deserialise_weights()
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	print("Evaluating in-domain on gsd corpus")

	print("On training data")
	train_data_gsd = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors_gsd = get_vectors_from_data(train_data_gsd)
	decision_corpus_gsd_train = get_decision_corpus(weights, train_vectors_gsd, tag_list)
	evaluate_accuracy(decision_corpus_gsd_train)

	print("On testing data")
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


	print("SRCMF data")
	test_data_SRCMF = get_data_from_file("./Eval_HorsDomaine/French_SRCMF/fro_srcmf-ud-test.conllu")
	test_vectors_SRCMF = get_vectors_from_data(test_data_SRCMF)
	decision_corpus_SRCMF = get_decision_corpus(weights, test_vectors_SRCMF, tag_list)
	evaluate_accuracy(decision_corpus_SRCMF)
	print()
