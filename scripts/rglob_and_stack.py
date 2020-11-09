#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import random
import os
import sys
import datetime
from functools import reduce







def recursive_rowstack_dataframes(basedir, filename, path_keyword=None):
	"""Combine all files found under this directory or its subdirectories that match this file name and return the stacked dataframe.
	
	Args:
		basedir (str): The base directory to recursively search under.
	
		filename (str): The final path component to use to look for compatible files.
	
		path_keyword (str): A string that has to be presented in the full path or that file is not used, optional.
	
	
	Returns:
		pandas.DataFrame: A dataframe resulting from stacking all files under that directory with the provided name.
	"""
	dfs = []
	for path in Path(basedir).rglob(filename):
		if (path_keyword == None) or (path_keyword in str(path)):
			dfs.append(pd.read_csv(path))
	if len(dfs)>1:
		df = pd.concat(dfs)
		df.reset_index(drop=True, inplace=True)
		return(df)
	else:
		return(None)
		
	




def columnstack_distances_dataframes(basedir, filename, path_keyword=None):
	"""Combine all files found under this directory or its subdirectories that match this file name and return the stacked dataframe.
	
	Args:
		basedir (str): The base directory to recursively search under.
	
		filename (str): The final path component to use to look for compatible files.
	
		path_keyword (str): A string that has to be presented in the full path or that file is not used, optional.
	
	
	Returns:
		pandas.DataFrame: A dataframe resulting from stacking all files under that directory with the provided name.
	"""
	dfs = []
	for path in Path(basedir).rglob(filename):
		if (path_keyword == None) or (path_keyword in str(path)):
			dfs.append(pd.read_csv(path))
	if len(dfs)>1:
		shared_columns = ["group_id", "full_name", "n"]
		df_final = reduce(lambda left,right: pd.merge(left,right,on=shared_columns, how="inner"), dfs)
		for df in dfs:
			assert df_final.shape[0] == df.shape[0]
		return(df_final)
	else:
		return(None)
		





	






# Read in the file that maps names used internally to names used in figures.
naming_dataframe_path = "/Users/irbraun/phenologs-with-oats/names.tsv"
name_df = pd.read_csv(naming_dataframe_path, sep="\t")
name_to_display_name = dict(zip(name_df["name_in_notebook"].values, name_df["name"]))
name_to_order = dict(zip(name_df["name_in_notebook"].values, name_df["order"]))





# Should we check paths for a keyword?
if len(sys.argv)>1:
	path_keyword = sys.argv[1]
else:
	path_keyword = None





# Create a new output folder to hold the stacked tables.
if path_keyword == None:
	OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}".format("stacked",datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
else:
	OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}_{}".format("stacked",datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999),path_keyword))
os.mkdir(OUTPUT_DIR)
	
	


# Where to look for these files to stack?
BASEDIR = "../outputs"




# The names of the final path components for files that should be stacked.
FILES_TO_BE_STACKED = [
	"approaches.csv",
	"auc.csv",
	"f1_max.csv",
	"f2_max.csv",
	"full_table_with_all_metrics.csv",
	"precision_recall_curves.csv",
	"histograms.csv",
	"correlations.csv",
	"all_pmn_only_within_distances_melted.csv",
	"all_kegg_only_within_distances_melted.csv",
	"all_subsets_within_distances_melted.csv",
	"curated_pmn_only_within_distances_melted.csv",
	"curated_kegg_only_within_distances_melted.csv",
	"curated_subsets_within_distances_melted.csv"
]
	
	


# For each of those files, stack them and write to a new file. Sort them if there is a column named "order".
for filename in FILES_TO_BE_STACKED:
	new_filename = "stacked_{}".format(filename)
	df = recursive_rowstack_dataframes(BASEDIR, filename, path_keyword)
	if isinstance(df, pd.DataFrame):

		if "name_key" in df.columns:
			all_old_columns = df.columns
			df["name_value"] = df["name_key"].map(name_to_display_name)
			df["order"] = df["name_key"].map(name_to_order)
			new_columns = ["name_value","order"]
			new_columns.extend([x for x in all_old_columns if x != "order"])
			df = df[new_columns]

		if "order" in df.columns:
			df.sort_values(by=["order"], inplace=True)
		df.to_csv(os.path.join(OUTPUT_DIR,new_filename), index=False)




print("finished vertically stacking output files")










# Ones that are stacked horizontally and require re-organization that's different than the above files.
FILES_TO_BE_STACKED = [
	"all_pmn_only_within_distances.csv",
	"all_kegg_only_within_distances.csv",
	"all_subsets_within_distances.csv",
	"curated_pmn_only_within_distances.csv",
	"curated_kegg_only_within_distances.csv",
	"curated_subsets_within_distances.csv",
]
	
for filename in FILES_TO_BE_STACKED:
	new_filename = "stacked_{}".format(filename)
	df = columnstack_distances_dataframes(BASEDIR, filename, path_keyword)
	if isinstance(df, pd.DataFrame):
		if "order" in df.columns:
			df.sort_values(by=["order"], inplace=True)
		df.to_csv(os.path.join(OUTPUT_DIR,new_filename), index=False)

print("finished horizontally stacking output files")




















