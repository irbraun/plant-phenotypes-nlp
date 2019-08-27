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



from phenolog.datasets.gene import Gene
from phenolog.nlp.preprocess import concatenate_descriptions, concatenate_with_bar_delim





class Dataset:


	def __init__(self):
		self.col_names = ["id", "species", "gene_names", "description", "terms", "reference"]
		self.col_names_without_id = self.col_names
		self.col_names_without_id.remove("id")
		self.df = pd.DataFrame(columns=self.col_names)



	def add_data(self, df_newlines):
		df_newlines = df_newlines[self.col_names_without_id]
		df_newlines["id"] = None
		df_newlines.fillna("", inplace=True)
		self.df = self.df.append(df_newlines, ignore_index=True, sort=False)
		self.df = self.df.drop_duplicates(keep="first", inplace=False)
		self.df.reset_index(drop=True, inplace=True)
		self._reset_ids()



	def randomly_subsample_dataset(self, n, seed):
		self.df = self.df.sample(n=n, random_state=seed)
		self._reset_ids()



	def _reset_ids(self):
		self.df.id = [str(i) for i in self.df.index.values]
		self.df = self.df[["id", "species", "gene_names", "description", "terms", "evidence", "reference"]]




	

	def get_gene_dictionary(self):
		gene_dict = {}
		for row in self.df.itertuples():
			delim = "|"
			gene_names = row.gene_names.split(delim)
			gene_obj = Gene(names=gene_names, species=row.species)
			gene_dict[row.id] = gene_obj
		return(gene_dict)

	
	def get_description_dictionary(self):
		description_dict = {identifier:description for (identifier,description) in zip(self.df.id, self.df.description)}
		return(description_dict)







	def set_genes_as_nodes(self):

		# Only perform this operation on slices of the data for one species at a time.
		# Enforces that genes with the same name across two species have to be two seperate nodes.
		# Nodes that correspond to different species can never be merged.
		num_new_rows = 0
		for species in pd.unique(self.df.species):
			print(species)


			# (1). Create a mapping from gene name strings to row indices where that string is mentioned.
			gene_mention_map = defaultdict(list)
			for row in self.df.itertuples():
				if row.species == species:
					delim = "|"
					gene_names = row.gene_names.split(delim)
					for gene_name in gene_names:
						gene_mention_map[gene_name].append(row.Index)


			# (2). Create a list of sets where all indices in a given set contain synonymous genes (overlap in >=1 name used).
			list_of_sets_of_row_indices = []
			for gene_name in gene_mention_map.keys():

				# Determine which existing synonymous set this gene belongs to, if any.
				set_index_where_this_gene_belongs = -1
				i = 0
				for set_of_row_indices in list_of_sets_of_row_indices:
					if len(set(gene_mention_map[gene_name]).intersection(set_of_row_indices)) > 0:
						set_index_where_this_gene_belongs = i
						list_of_sets_of_row_indices[i].update(gene_mention_map[gene_name])
						break
					i = i+1

				# If this gene doesn't belong in any of those sets, start a new set with all it's corresponding row indices.
				if set_index_where_this_gene_belongs == -1:
					list_of_sets_of_row_indices.append(set(gene_mention_map[gene_name]))


			# (3). Add rows which contain merged information from multiple rows where the same gene was mentioned.
			num_new_rows = num_new_rows + len(list_of_sets_of_row_indices)
			for set_of_row_indices in list_of_sets_of_row_indices:
				relevant_rows = self.df.iloc[list(set_of_row_indices)]
				description = concatenate_descriptions(*relevant_rows.description.tolist())
				gene_names = concatenate_with_bar_delim(*relevant_rows.gene_names.tolist())
				pmids = concatenate_with_bar_delim(*relevant_rows.pmid.tolist())
				new_row = {
					"id":None,
					"species":species,
					"description":description,
					"gene_names":gene_names,
					"pmid":pmids,
				}
				self.df.append(new_row, ignore_index=True, sort=False)


		# Retain only the newly added rows, reset the ID values for each row that correspond to one node.
		self.df = self.df.iloc[-num_new_rows:]
		self.df.reset_index(drop=True, inplace=True)
		self._reset_ids()









	def describe(self):
		print("Number of rows in the dataframe: {}".format(len(self.df)))
		print("Number of unique IDs:            {}".format(len(pd.unique(self.df.id))))
		print("Number of unique descriptions:   {}".format(len(pd.unique(self.df.description))))
		print("Number of unique gene name sets: {}".format(len(pd.unique(self.df.gene_names))))
		print("Number of species represented:   {}".format(len(pd.unique(self.df.species))))




