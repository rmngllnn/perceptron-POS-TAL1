
from perceptron import * 
#from perceptron import tag_list
#from perceptron import train_vectors

import pickle

def serialise_weights():
	"""Saves the weights in a pickle file to prevent from having to
	retrain the perceptron every single time.

	weights: the weights to be serialized, as created/formatted by train()
	"""
    	
	weights = train(train_vectors, tag_list, MAX_EPOCH = 10, evaluate_epochs = False, dev_vectors = None)
	file_name = "weights.pkl"
	file = open(file_name, "wb")
	pickle.dump(weights, file)
	file.close()

if "__main__" == __name__:
	
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	train_data_gsd = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors_gsd = get_vectors_from_data(train_data_gsd)
	vocabulary = get_vocabulary(train_vectors_gsd)

	weights = train(train_vectors_gsd, tag_list)
	serialise_weights(weights)