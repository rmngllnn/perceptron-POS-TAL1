import random
from perceptron_serialise import serialise_weights
from perceptron_basics import *

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



def train(train_vectors, tag_list, max_epoch = MAX_EPOCH, evaluate_epochs = False, dev_vectors = None):
	"""Creates and return the weights to score each tags and predict the best 
	one. Weights are averaged by adding each temporary value of them to each
	other.
	If evaluate_epochs = True, evaluates at every epoch to find the best
	MAX_EPOCH (up to the value given as parameter of the function).

	train_vectors: list of tuples (vector_word, gold_POS), as created/formatted 
	by get_vectors_from_data, to get the weights
	tag_list: list of existing tags
	evaluate_epochs: whether to evaluate the weights at each epoch
	dev_vectors: same, but a separate set, to evaluate the accuracy of the 
	weights depending on the number of epochs they were trained on
	"""

	average = {}
	if evaluate_epochs:
		accuracy_by_epoch = {}
		print("For MAX_EPOCH in "+str(range(0,max_epoch))+"\nn_epochs\taccuracy")

	for epoch in range(0, max_epoch):
		print("epoch: "+str(epoch+1))
		weights = {}
		random.shuffle(train_vectors)
		index_word = 0
		n_words = len(train_vectors)

		for word in train_vectors:
			index_word += 1
			predicted_tag = predict_tag(word[0], weights, tag_list)
			gold_tag = word[1]
			if predicted_tag != gold_tag:
				add_vector_to_weights(word[0], weights, predicted_tag, -1)
				add_vector_to_weights(word[0], weights, gold_tag, +1)
		"""In the pseudo code in the lesson, the following line is placed inside the loop above.
		As it dramatically increased training time, we left it in the outer loop.
		As such, it averages between each epochs instead of each word."""
		add_weights_to_average(average, weights)

		if evaluate_epochs:
			accuracy_by_epoch[epoch] = evaluate_accuracy(get_decision_corpus(weights, dev_vectors, tag_list))

	if evaluate_epochs:
		print("best MAX_EPOCH: "+str(max(results, key=lambda n_epochs: (accuracy_by_epoch[epoch], epoch))))

	return average



if "__main__" == __name__:
	"""Creates, trains and saves an averaged POS-tagging perceptron.
	"""
	
	tag_list = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
			 "PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	train_data_gsd = get_data_from_file("./fr_gsd-ud-train.conllu")
	train_vectors_gsd = get_vectors_from_data(train_data_gsd)

	"""
	dev_data = get_data_from_file("./fr_gsd-ud-dev.conllu")
	dev_vectors = get_vectors_from_data(dev_data)
	weights = train(dev_vectors, tag_list, MAX_EPOCH=50, evaluate_epochs=True, dev_vectors=dev_vectors)
	"""

	weights = train(train_vectors_gsd, tag_list)
	serialise_weights(weights)
