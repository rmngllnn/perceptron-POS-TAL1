#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

"""This file is dealing with the serialisation of weight vectors
   You don‘t have to run this file, the functions are used in other files"""


def serialise_weights(weights):
	"""Saves the weights in a pickle file to prevent from having to
	retrain the perceptron every single time.

	weights: the weights to be serialized, as created/formatted by train()
	"""
    
	file_name = "weights.pkl"
	file = open(file_name, "wb")
	pickle.dump(weights, file)
	file.close()


def deserialise_weights():
	"""Loads the weights from the serialised file on desktop.
	"""
    
	data = open("weights.pkl", 'rb') 
	weights = pickle.load(data)
	data.close()

	return weights



