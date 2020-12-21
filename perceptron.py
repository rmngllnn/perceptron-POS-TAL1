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

def get_word_vector(sentence, word, index):
	"""Calculates the features of a given word, returns that vector.
	Hypothesis: word, word before, word after, pre- and suffixes with len from 1 to 3, are enough to determine POS tagging. RAF
	sentence: list of Strings (words), the original sentence.
	word: String, the word.
	index: index of word in the sentence. Taken from the corpus, so index of sentence[0] is 1.
	"""
	
	vector = {}
	len_word = len(word)
	len_sentence = len(sentence)
	index -= 1 #to get the index in the list

	vector["word="+word] = 1

	if index == 0:
		vector["is_first"] = 1
	else:
		vector["word-1="+sentence[index-1]] = 1

	if index == len_sentence -1:
		vector["is_last"] = 1
	else:
		vector["word+1="+sentence[index+1]] = 1
		
	if word[0].upper() == word[0]:
		vector["is_capitalized"] = 1
	if word.upper() == word:
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



def predict_tag(vector, weights, tagset):
	"""Predicts and returns the tag of a word.
	vector: features of the word, as calculated/formatted by get_word_vector
	weights: the weights for each feature in the prediction
	tagset: possible tags
	"""
	
	scores = {}
	for tag in tagset:
		scores[tag] = 0
		for feature in weights:
			weights_feature = weights[feature]
			if feature in vector:
				scores[tag] += weights_feature.get(tag,0) * vector[feature]
	return max(scores, key=lambda tag: (scores[tag], tag))



def add_vector_to_weights(vector, weights, tag, factor):
	"""Adds or substract (for factor = 1 and -1) a vector to a set of weights. Auxiliary fonction for train.
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
	"""Auxiliary fonction of train, adds the calculated weights to the average, to smooth variations out and allow for reaching a limit.
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



def train(vectors_corpus, tagset, MAX_EPOCH = 1):
	
	"""creation of the weights to score the tags when predicting the best one, 
	"""
	average = {}

	for epoch in range(0, MAX_EPOCH):
		print("epoch: "+str(epoch))
		weights = {}
		random.shuffle(vectors_corpus)
		index_word = 0
		n_words = len(vectors_corpus)
		for word in vectors_corpus:
			#print("epoch "+str(epoch)+"/"+str(MAX_EPOCH)+": "+str(index_word)+"/"+str(n_words)+" words") #TBD
			index_word += 1
			predicted_tag = predict_tag(word[0], weights, tagset)
			gold_tag = word[1]
			if predicted_tag != gold_tag:
				add_vector_to_weights(word[0], weights, predicted_tag, -1)
				add_vector_to_weights(word[0], weights, gold_tag, +1)
		add_weights_to_average(average, weights)
	#print(average) # TBD
	return average



def get_data_from_file(file = "./fr_gsd-ud-train.conllu"):
	
	"""extraction des données à partir des fichiers conllu
	   return : list de dictionnaires 
       (exemple pour une phrase : 
		[{index du mot dans la phrase : 1, mot : mot1, gold_pos : ADV}, 
         {index du mot dans la phrase 2, mot : mot2, gold_pos : DET}], etc."""
	
	data = [] # list of lists (sentences) of dictionaries (words)

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
	
	"""Création des vecteurs à partir des données extraites du fichiers
	   return : list of tuples""" 
	
	# vectors: list of tuples, with (word vector, gold POS tag)
	vectors = []
	for sentence_data in data:
		sentence = []
		for word_data in sentence_data:
			sentence.append(word_data["word"])
		for word_data in sentence_data:
			word_vector = get_word_vector(sentence, word_data["word"], word_data["index"])
			to_append = (word_vector, word_data["gold_POS"])
			vectors.append(to_append)
	return vectors


if "__main__" == __name__:
	
	start_time = time.time()
	
	#train_data = get_data_from_file("./fr_gsd-ud-train.conllu")
	dev_data = get_data_from_file("./fr_gsd-ud-dev.conllu")
	
	#test_data = get_data_from_file("./fr_gsd-ud-test.conllu")

	#train_vectors = get_vectors_from_data(train_data)
	dev_vectors = get_vectors_from_data(dev_data)
	#test_vectors = get_vectors_from_data(test_data)
	

	tagset = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	weigths = {} #donc que des 0 
	weights = train(dev_vectors, tagset) #weights[feature][tag]
	
	print(int(time.time()-start_time), " secondes")

	#To do : validation with MAX_EPOCH and dev_vectors
	#To do : evaluation with test_vectors
