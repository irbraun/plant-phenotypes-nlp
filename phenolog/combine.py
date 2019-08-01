from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from sklearn.linear_model import LinearRegression





def combine_with_mean(dfs_dict):
	consistent = _verify_dfs_are_consistent(*dfs_dict.values())
	print(consistent)
	df = pd.concat(dfs_dict.values(), ignore_index=True)
	df = df.groupby(["from","to"], sort=False).mean().reset_index()
	df.columns = ["from", "to", "similarity"]
	return(df)

	



def combine_with_weights(dfs_dict, weights_dict):
	for name,df in dfs_dict.items():
		df["name"] = name
	consistent_1 = _verify_dfs_are_consistent(*dfs_dict.values())
	consistent_2 = (len(set(dfs_dict.keys()).difference(set(weights_dict.keys()))) == 0)
	print(consistent_1)
	print(consistent_2)
	df = pd.concat(dfs_dict.values(), ignore_index=True)
	df = df.groupby(["from","to"], sort=False).apply(_get_y, weights_dict=weights_dict).reset_index()
	df.columns = ["from", "to", "similarity"]
	return(df)



def _get_y(data, weights_dict):
	weighted_values = [sim*weights_dict[name] for (sim,name) in zip(data["similarity"].values, data["name"].values)]
	product = np.nanprod(weighted_values)
	return(product)












def combine_with_linear_model(dfs_dict, regression_model):

	for name,df in dfs_dict.items():
		df["name"] = name
	names = list(dfs_dict.keys())
	consistent = _verify_dfs_are_consistent(*dfs_dict.values())
	print(consistent)
	df = pd.concat(dfs_dict.values(), ignore_index=True)
	df = df.groupby(["from","to"], sort=False).apply(_get_y_reg, reg=regression_model, name_list=names).reset_index()
	df.columns = ["from", "to", "similarity"]
	return(df)

def _get_y_reg(data, reg, name_list):
	# Need to ensure that the order of the features is the same as in the learn_weights thing below.
	feature_set = [data[data["name"]==name]["similarity"].values[0] for name in name_list]
	# The reshape is necessary because this just a single sample (with k features), but based on training this model can take a list of lists
	x = np.array(feature_set).reshape(1, -1)
	# This is list of 1 predcitions, we only need the first one.
	y = reg.predict(x)
	y = y[0]
	return(y)









def learn_weights_linear_regression(dfs_dict, target_df):
	"""
	The similarity dataframes have the generated values of similarity from each of the 
	methods used. The target dataframe some subset of the same object IDs (in the 'from' 
	and 'to' columns) but the values in the similarity columns are floats which refer to
	the target value that the model should try and predict given the set of dataframes
	that were provided, one for each method. The thing that is returned should be the 
	weights dictionary with names mapping to weight values (the coefficients for each 
	method).
	"""
	

	# TODO there is a problem with NaN here.


	from sklearn.linear_model import LinearRegression


	# Generate lists of feature value lists, in the shape we need for X.
	names = list(dfs_dict.keys())
	feature_sets = []
	target_values = []
	for (f,t) in zip(target_df["from"].values, target_df["to"].values):
		feature_set = []
		for name in names:
			similarity_value = dfs_dict[name][(dfs_dict[name]["from"]==f) & (dfs_dict[name]["to"]==t)]["similarity"].values[0]
			feature_set.append(similarity_value)
		feature_sets.append(feature_set)
		target_values.append(target_df[(target_df["from"]==f) & (target_df["to"]==t)]["similarity"].values[0])


	# Data structures needed for fitting the model.
	X = np.array(feature_sets)
	y = np.array(target_values)

	reg = LinearRegression().fit(X, y)
	print(reg.score(X,y))
	print(reg.coef_)
	print(reg.intercept_)
	return(reg)








	'''
	>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
	>>> # y = 1 * x_0 + 2 * x_1 + 3
	>>> y = np.dot(X, np.array([1, 2])) + 3
	>>> reg = LinearRegression().fit(X, y)
	>>> reg.score(X, y)
	1.0
	>>> reg.coef_
	array([1., 2.])
	>>> reg.intercept_ 
	3.0000...
	>>> reg.predict(np.array([[3, 5]]))
	array([16.])
	'''


















def _verify_dfs_are_consistent(*similarity_dfs):
	id_sets = [set() for i in range(0,len(similarity_dfs))]
	for i in range(0,len(similarity_dfs)):
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["from"].values)))
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["to"].values)))
	for (s1, s2) in list(itertools.combinations_with_replacement(id_sets, 2)):	
		if not len(s1.difference(s2)) == 0:
			return(False)
	return(True)








