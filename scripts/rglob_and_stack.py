#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import random
import os
import datetime







def recursive_rowstack_dataframes(basedir, filename):
    """Combine all files found under this directory or its subdirectories that match this file name and return the stacked dataframe.
    
    Args:
        basedir (str): The base directory to recursively search under.
        
        filename (str): The final path component to use to look for compatible files.
    
    Returns:
        pandas.DataFrame: A dataframe resulting from stacking all files under that directory with the provided name.
    """
    dfs = []
    for path in Path(basedir).rglob(filename):
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return(df)
        
    
    




# Create a new output folder to hold the stacked tables.
OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}".format("stacked",datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
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
]
    
    








# For each of those files, stack them and write to a new file. Sort them if there is a column named "order".
for filename in FILES_TO_BE_STACKED:
    new_filename = "stacked_{}".format(filename)
    df = recursive_rowstack_dataframes(BASEDIR, filename)
    if "order" in df.columns:
        df.sort_values(by=["order"], inplace=True)
    df.to_csv(os.path.join(OUTPUT_DIR,new_filename), index=False)

    
print("finished stacking output files")



