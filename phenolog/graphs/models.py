from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
from functools import reduce
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier








def apply_weights(df, predictor_columns, weights_dict):
	"""
	Generates a dataframe with a single value column calculated from combining a
	number of the old columns with the provided weights. Checks to make sure that
	the weights refer to columns which exist in the passed in dataframe.

	Args:
	    df (pandas.DataFrame): A dataframe with predictor columns to be combined.
	    predictor_columns (list): The names of the predictor columns in a list.
	    weights_dict (dict): Mapping from predictor column names to their weight.
	
	Returns:
	    pandas.DataFrame: The resulting dataframe with columns combined.
	"""
	# Check to make sure the arguments are compatible with each other.
	if not len(set(predictor_columns).difference(set(weights_dict.keys()))) == 0:
		raise Error("Names in the weight dictionary don't match list of predictors.")
	X = _get_X(df, predictor_columns)
	w = np.array([weights_dict[name] for name in predictor_columns])
	W = np.tile(w, (X.shape[0], 1))
	multiplied = np.multiply(X,W)
	y = [np.sum(row) for row in multiplied]
	df["similarity"] = y
	df = df[["from", "to", "similarity"]]
	return(df)




def apply_mean(df, predictor_columns):
	weight_for_all = 1.000 / float(len(predictor_columns))
	weights_dict = {name:weight_for_all for name in predictor_columns}
	return(apply_weights(df, predictor_columns, weights_dict))





def apply_linear_regression_model(df, predictor_columns, model):
	X = _get_X(df, predictor_columns)
	y = model.predict(X)
	df["similarity"] = y
	df = df[["from", "to", "similarity"]]
	return(df)




def apply_random_forest_model(df, predictor_columns, model, positive_label=1):
	X = _get_X(df, predictor_columns)
	class_probabilities = model.predict_proba(X)
	positive_class_label = positive_label
	positive_class_index = model.classes_.tolist().index(positive_class_label)
	positive_class_probs = [x[positive_class_index] for x in class_probabilities]
	df["similarity"] = positive_class_probs
	df = df[["from", "to", "similarity"]]
	return(df)








def train_linear_regression_model(df, predictor_columns, target_column):
	X,y = _get_X_and_y(df, predictor_columns, target_column) 
	reg = LinearRegression().fit(X, y)
	training_rmse = reg.score(X,y)
	coefficients = reg.coef_
	intercept = reg.intercept_ 
	return(reg)




def train_random_forest_model(df, predictor_columns, target_column, num_trees=100, function="gini", max_depth=None, seed=None):
	X,y = _get_X_and_y(df, predictor_columns, target_column)
	rf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, criterion=function, random_state=seed)
	rf.fit(X,y)
	feature_importance = rf.feature_importances_
	return(rf)
















def combine_dfs_with_name_dict(dfs_dict):
	"""Produce a dataframe in a shape that can be used to train models.
	Args:
	    dfs_dict (dict): Mapping from name strings to dataframe objects.
	Returns:
	    pandas.DataFrame: A single dataframe where each similarity column
	    in the original dataframes has become a new column in the combined
	    dataframe where the header for that column is the name that was in
	    the dictionary. This way each individual passed in dictionary can
	    be used to collect named features for a new dataframe that can be 
	    used to train models for predicting the best combination of features.
	"""
	for name,df in dfs_dict.items():
		df.rename(columns={"similarity":name}, inplace=True)
	_verify_dfs_are_consistent(*dfs_dict.values())
	merged_df = reduce(lambda left,right: pd.merge(left,right,on=["from","to"], how="outer"), dfs_dict.values())
	merged_df.fillna(0.000, inplace=True)
	return(merged_df)







def _verify_dfs_are_consistent(*similarity_dfs):
	"""Check that each dataframe specifies the same set of edges.
	Args:
	    *similarity_dfs: Any number of dataframe arguments.
	Raises:
	    Error: The dataframes were found to not all be describing the same graph.
	"""
	id_sets = [set() for i in range(0,len(similarity_dfs))]
	for i in range(0,len(similarity_dfs)):
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["from"].values)))
		id_sets[i].update(list(pd.unique(similarity_dfs[i]["to"].values)))
	for (s1, s2) in list(itertools.combinations_with_replacement(id_sets, 2)):	
		if not len(s1.difference(s2)) == 0:
			raise Error("Dataframes specifying networks are not consisent.")





def _get_X_and_y(df, predictor_columns, target_column):
	"""Get arrays for X and y from the dataframe.
	"""
	feature_sets = df[predictor_columns].values.tolist()
	target_values = df[target_column].values.tolist()
	X = np.array(feature_sets)
	y = np.array(target_values)
	return(X,y)



def _get_X(df, predictor_columns):
	"""Get array for X from the dataframe.
	"""
	feature_sets = df[predictor_columns].values.tolist()
	X = np.array(feature_sets)
	return(X)



















