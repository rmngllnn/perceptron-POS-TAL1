# corpus:
# https://universaldependencies.org/treebanks/fr_gsd/index.html#ud-french-gs
# https://github.com/UniversalDependencies/UD_French-GSD

# bibliography: 
# https://moodle.u-paris.fr/pluginfile.php/649525/mod_resource/content/1/perceptron.pdf (training pseudo-code algorithm)
# https://www.nltk.org/_modules/nltk/tag/perceptron.html
# https://www.guru99.com/pos-tagging-chunking-nltk.html
# https://becominghuman.ai/part-of-speech-tagging-tutorial-with-the-keras-deep-learning-library-d7f93fa05537 (list of features)



def create_word_vector(sentence, word, index): #hypothesis: word, word before, word after, pre- and suffixes with len from 1 to 3, are enough to determine POS tagging
	vector = {}
	if index == 0:
		vector["is_first"] = 1
	if index == len(sentence) -1:
		vector["is_last"] = 1
	if word[0].upper() == word[0]:
		vector["is_capitalized"] = 1
	if word.upper() == word:
		vector["is_uppercase"] = 1

	word_minus_1 = ""
	if index != 0:
		word_minus_1 = sentence[index-1]
	vector["word-1="+word_minus_1] = 1

	vector["word="+word] = 1

	word_plus_1 = ""
	if index > len(sentence) - 1:
		word_plus_1 = sentence[index+1]
	vector["word-1="+word_plus_1] = 1

	#RAF determine if "", then still a feature, or if then = 0
	prefix_1 = ""
	if len(word) > 0:
		prefix_1 = word[0:1]
	vector["prefix1="+prefix_1] = 1

	prefix_2 = ""
	if len(word) > 1:
		prefix_1 = word[0:2]
	vector["prefix2="+prefix_2] = 1

	prefix_3 = ""
	if len(word) > 2:
		prefix_3 = word[0:3]
	vector["prefix3="+prefix_3] = 1

	suffix_1 = ""
	if len(word) > 0:
		suffix_1 = word[0:1]
	vector["suffix1="+suffix_1] = 1

	suffix_2 = ""
	if len(word) > 1:
		suffix_2 = word[0:2]
	vector["suffix2="+suffix_2] = 1

	suffix_3 = ""
	if len(word) > 2:
		suffix_3 = word[0:3]
	vector["suffix3="+suffix_3] = 1

	return vector



def predict_tag(vector, weights, tagset):
	scores = {}
	for tag in tagset:
		scores[tag] = 0
		for feature in weights:
			if feature in vector:
				scores[tag] += weights[feature][tag] * vector[feature]
	return max(scores, key=lambda tag: (scores[tag], tag))



def add_vector_to_weight(vector, weights, tag, factor):
	for feature in vector:
		weights[feature][tag] = weights[feature].get(tag, 0) + vector[feature]*factor



def add_weight_to_average(average, weight):
	for feature in weight:
		for tag in weight[feature]:
			average[feature][tag] = average[feature].get(tag, 0) + weight[feature][tag]



def train(corpus, tagset, MAX_EPOCH = 100):
	#RAF formatting of training corpus
	average = {}
	for epoch in range(0, MAX_EPOCH):
		weight = {}
		#shuffle corpus
		for word in corpus:
			predicted = predict_tag(vector, index, tagset)
			if predicted != gold:
				add_vector_to_weight(vector, weight, predicted, -1)
				add_vector_to_weight(vector, weight, gold, +1)
			add_weight_to_average(average, weight)
	return average



def get_data(file = "./fr_gsd-ud-dev.conllu"):
    data = []
    with open(file, "r") as raw:
        for line in raw.readlines():
            #RAFFFFF
    return data



if "__main__" == __name__:
	train_set = get_data("./fr_gsd-ud-train.conllu")
	dev_set = get_data("./fr_gsd-ud-dev.conllu")
	test_set = get_data("./fr_gsd-ud-test.conllu")

	tagset = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

	#weigths = {} #donc que des 0 
	#weights = train(train_set, tagset) #weights[feature][tag]

	#RAF validation with MAX_EPOCH
	#RAF evaluation
