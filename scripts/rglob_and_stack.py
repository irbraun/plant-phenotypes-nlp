#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import random
import os
import sys
import datetime







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
	"full_table_with_all_metrics.csv",
	"precision_recall_curves.csv",
	"histograms.csv",
	"correlations.csv"
]
	
	




# For each of those files, stack them and write to a new file. Sort them if there is a column named "order".
for filename in FILES_TO_BE_STACKED:
	new_filename = "stacked_{}".format(filename)
	df = recursive_rowstack_dataframes(BASEDIR, filename, path_keyword)
	if isinstance(df, pd.DataFrame):
		if "order" in df.columns:
			df.sort_values(by=["order"], inplace=True)
		df.to_csv(os.path.join(OUTPUT_DIR,new_filename), index=False)

	
print("finished stacking output files")



