import sys
import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.simplefilter('ignore')
sys.path.append("../../oats")
from oats.datasets.dataset import Dataset
from oats.datasets.known import Known
from oats.utils.utils import save_to_pickle, load_from_pickle



# Define the paths to files used to make the training and testing split.
ppn_edgelist_path = "../data/supplemental_files_oellrich_walls/13007_2015_53_MOESM9_ESM.txt"
complete_dataset_pickle_path = "../data/pickles/full_dataset.pickle"
training_pickle_path = "../data/pickles/training_dataset.pickle" 
testing_pickle_path = "../data/pickles/testing_dataset.pickle"



# Subset the complete dataset to only include genes that are in the Oellrich, Walls paper.
complete_dataset = load_from_pickle(complete_dataset_pickle_path)
complete_dataset.filter_has_description()
known = Known(complete_dataset.get_name_to_id_dictionary(), ppn_edgelist_path)
complete_dataset.filter_with_ids(known.ids)
save_to_pickle(obj=complete_dataset, path=testing_pickle_path)



# Subset the complete dataset to only include all the rest of the genes.
complete_dataset = load_from_pickle(complete_dataset_pickle_path)
complete_dataset.filter_has_description()
known = Known(complete_dataset.get_name_to_id_dictionary(), ppn_edgelist_path)
training_ids = list(set(complete_dataset.get_ids()) - set(known.ids))
complete_dataset.filter_with_ids(training_ids)
save_to_pickle(obj=complete_dataset, path=training_pickle_path)