import sys
import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.simplefilter('ignore')
sys.path.append("../../oats")
from oats.biology.dataset import Dataset
from oats.utils.utils import save_to_pickle








def save_combined_dataset_to_pickle(pickle_path, input_dir, *input_filenames):

	print("\nworking on creating", os.path.basename(pickle_path))
	dataset = Dataset()
	for filename in input_filenames:
		filepath = os.path.join(input_dir, filename)
		dataset.add_data(pd.read_csv(filepath, lineterminator="\n"))
		print("finished adding data from {}".format(filepath))

	# Saving a version of the dataset prior to merging based on gene names.
	split_basename = os.path.basename(pickle_path).split(".")
	unmerged_pickle_path = os.path.join(os.path.dirname(pickle_path), "{}_unmerged.{}".format(split_basename[0],split_basename[1]))
	save_to_pickle(obj=dataset, path=unmerged_pickle_path)	

	# Saving the version of the dataset after merging based on gene names.
	print("merging rows based on gene names...")
	dataset.collapse_by_all_gene_names()
	save_to_pickle(obj=dataset, path=pickle_path)
	print("done")





reshaped_dir = "../data/reshaped_files"
#all_csv_files = [os.path.basename(f) for f in glob.iglob(os.path.join(reshaped_dir,"*.csv"))]
#save_combined_dataset_to_pickle("../data/pickles/full_dataset.pickle", reshaped_dir, *all_csv_files)
save_combined_dataset_to_pickle("../data/pickles/gene_phenotype_dataset_all_text.pickle", reshaped_dir, "ppn_phenes.csv", "ppn_phenotypes.csv", "sly_phenotypes.csv", "zma_phenotypes.csv", "ath_phenotypes.csv")
save_combined_dataset_to_pickle("../data/pickles/gene_phenotype_dataset_all_text_and_annotations.pickle", reshaped_dir, "ppn_phenes.csv", "ppn_phenotypes.csv", "ppn_annotations.csv", "sly_phenotypes.csv", "zma_phenotypes.csv", "ath_phenotypes.csv", "ath_high_confidence_go_annotations.csv", "ath_high_confidence_po_annotations.csv", "zma_high_confidence_go_annotations.csv")
save_combined_dataset_to_pickle("../data/pickles/gene_phenotype_dataset_ppn_text_and_annotations.pickle", reshaped_dir, "ppn_phenes.csv", "ppn_phenotypes.csv", "ppn_annotations.csv")

