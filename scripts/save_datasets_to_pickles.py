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








def save_combined_dataset_to_pickle(pickle_path, input_dir, *input_filenames):
	dataset = Dataset()
	for filename in input_filenames:
		filepath = os.path.join(input_dir, filename)
		dataset.add_data(pd.read_csv(filepath, lineterminator="\n"))
		print("finished adding data from {}".format(filepath))
	print("merging rows based on gene names...")
	dataset.collapse_by_all_gene_names()
	print("done merging")
	save_to_pickle(obj=dataset, path=pickle_path)






reshaped_dir = "../data/reshaped_files"
save_combined_dataset_to_pickle("../data/pickles/full_dataset.pickle", reshaped_dir, glob.iglob(os.path.join(reshaped_dir,"*.csv")))
save_combined_dataset_to_pickle("../data/pickles/ppn_dataset.pickle", reshaped_dir, "ppn_annotations.csv", "ppn_phenes.csv", "ppn_phenes.csv")
save_combined_dataset_to_pickle("../data/pickles/ath_go_dataset.pickle", reshaped_dir, "ath_high_confidence_go_annotations.csv", "ath_phenotypes.csv")
save_combined_dataset_to_pickle("../data/pickles/ath_po_dataset.pickle", reshaped_dir, "ath_high_confidence_po_annotations.csv", "ath_phenotypes.csv")

