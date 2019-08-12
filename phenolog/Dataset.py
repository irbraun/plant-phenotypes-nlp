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
from pandas import DataFrame
import os
import sys
import glob



from phenolog.Gene import Gene





class Dataset:







	def __init__(self):
		self.col_names = ["id", "species", "description", "pubmed", "gene_names", "gene_ncbi", "gene_uniprot"]
		self.col_names_without_id = self.col_names
		self.col_names_without_id.remove("id")
		self.df = pd.DataFrame(columns=self.col_names)



	def add_data(self, df_newlines):
		df_newlines = df_newlines[self.col_names_without_id]
		df_newlines["id"] = None
		self.df = self.df.append(df_newlines, ignore_index=True, sort=False)
		self.df = self.df.drop_duplicates(keep="first", inplace=False)
		self._reset_ids()


	def subsample_data(self, num_to_retain):
		self.df = self.df.sample(n=num_to_retain)
		self._reset_ids()


	def _reset_ids(self):
		self.df["id"] = [str(i) for i in self.df.index.values]





	def get_description_dictionary(self, all_rows):
		# Every entry in the dataset is a key.
		if all_rows == 1:
			description_dict = {identifier:description for (identifier,description) in zip(self.df.id,self.df.description)}
			return(description_dict)
		# Every locus in the dataset is a key.
		if all_rows == 0:
			description_dict = {}
			for locus in pd.unique(self.df.locus):
				descriptions = self.df[self.df.locus == locus].description.values
				descriptions = [self._add_end_tokens(desc) for desc in descriptions]
				description = " ".join(description_dict).strip()
				description_dict[locus] = description
			return(description_dict)

	def _add_end_tokens(self, description):
		last_character = description[len(description)-1]
		end_tokens = [".", ";"]
		if not last_character in end_tokens:
			description = description+"."
		return(description)




	# There are a bunch of different ways to do this
	# figure out how to shrink the dataset when genes overlap etc.


	def get_gene_dictionary(self):
		gene_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			gene_names = row.gene_names.split(delim)
			gene_obj = Gene(names=gene_names, uniprot_id=row.gene_uniprot, ncbi_id=row.gene_ncbi)
			gene_dict[row.id] = gene_obj
		return(gene_dict)








	def describe(self):
		print("Number of rows in the dataframe:", len(self.df))
		print("Number of unique IDs:", len(pd.unique(self.df.id)))
		print("Number of unique descriptions:", len(pd.unique(self.df.description)))
		print("Number of unique loci:", len(pd.unique(self.df.locus)))
		print("Number of species represented:", len(pd.unique(self.df.species)))




