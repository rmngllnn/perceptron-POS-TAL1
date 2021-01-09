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
		vector["suffix1="+word[-1::]] = 1
	if len_word > 1:
		vector["prefix2="+word[0:2]] = 1
		vector["suffix2="+word[-2::]] = 1
	if len_word > 2:
		vector["prefix3="+word[0:3]] = 1
		vector["suffix3="+word[-3::]] = 1
	if len_word > 3:
		vector["prefix3="+word[0:4]] = 1
		vector["suffix3="+word[-4::]] = 1


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
