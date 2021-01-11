from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from perceptron_basics import *
from perceptron_serialisation import deserialise_weights


"""This program calculates the confusion matrix and the msot frequent confusions, using the corpus
specified in the main function and the weights serialized by the train program.
Edit the file path to the corpus and the name of the graph to be created in the main function."""



def get_confusion_matrix(decision_corpus, tag_list, graph_title):
	"""Calculates and returns a confusion matrix to analyse tagging performance. Also plots
	and saves it using plot_and_save_confusion_matrix().

	decision_corpus: list of tagging decisions, saved as dictionaries (word_vec, pred_pos,
	gold_pos), as created/formatted by get_decision_corpus()
	tag_list: list of existing tags
	graph_title: string, title of the heatmap to create and save
	"""

	matrix = np.zeros((len(tag_list), len(tag_list)))
	
	gold_pos = []
	pred_pos = []
	
	for decision in decision_corpus:
 		pred_pos.append(decision["predicted_tag"])
 		gold_pos.append(decision["gold_tag"])
	
	matrix += confusion_matrix(gold_pos, pred_pos, labels = tag_list)
	plot_and_save_confusion_matrix(matrix, tag_list, graph_title)

	return matrix



def plot_and_save_confusion_matrix(confusion_matrix, tag_list, graph_title):
	"""Prints and saves on desktop a heatmap of the confusion matrix.

	confusion_matrix: confusion matrix, as calculated by get_confusion_matrix()
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


	
def get_most_frequent_confusions(matrix):
	"""Returns the name of the 3 most frequent confusions.

	matrix : confusion matrix, as calculated by matrix_confusion()
	"""

	list_freq= []

	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if len(list_freq) == 3:
				if i != j and matrix[i][j] > list_freq[0]:
					list_freq.remove(list_freq[0])
					list_freq.append(matrix[i][j])
			elif i != j:
				list_freq.append(matrix[i][j])
				list_freq.sort()
	return list_freq



def print_errors(decision_corpus):
	"""Prints the incorrect decisions in order to be able to analyze them manually.

	decision_corpus: list of tagging decisions, saved as dictionaries (word_vec, pred_pos,
	gold_pos), as created/formatted by get_decision_corpus()
	"""

	for decision in decision_corpus:
		if decision["gold_tag"] != decision["predicted_tag"]:
			print(decision)



if "__main__" == __name__:
	"""Saves a confusion matrix graph and prints the three most frequent confusion types for the serialised
	perceptron, for the corpus specified in the code. Can also print the errors.
	"""

	weights = deserialise_weights()

	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	data = get_data_from_file("./Eval_HorsDomaine/French_SRCMF/fro_srcmf-ud-test.conllu") #to change depending on the preferred corpus
	vectors = get_vectors_from_data(data)
	decision_corpus = get_decision_corpus(weights, vectors, tag_list)

	matrix = get_confusion_matrix(decision_corpus, tag_list, "confusion_test_SRCMF") #same
	#print_errors(matrix)

	get_most_frequent_confusions(matrix)
