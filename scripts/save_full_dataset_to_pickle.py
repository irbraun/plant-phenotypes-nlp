import sys
import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.simplefilter('ignore')

sys.path.append("../../oats")
from oats.datasets.dataset import Dataset
from oats.utils.utils import save_to_pickle


dataset = Dataset()
reshaped_files_dir = "../data/reshaped_files"
for filepath in glob.iglob(os.path.join(reshaped_files_dir,"*.csv")):
	dataset.add_data(pd.read_csv(filepath, lineterminator="\n"))
	print("finished adding data from {}".format(filepath))
print("merging rows based on gene names...")
dataset.collapse_by_all_gene_names()
print("done merging")
save_to_pickle(obj=dataset, path="../data/pickles/full_dataset.pickle")