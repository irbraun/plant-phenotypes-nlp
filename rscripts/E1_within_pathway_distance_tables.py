#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import random
import os
import sys
import datetime
import glob
import subprocess
import math



# Read in the file that maps names used internally to names used in figures.
naming_dataframe_path = "/Users/irbraun/phenologs-with-oats/names.tsv"
name_df = pd.read_csv(naming_dataframe_path, sep="\t")
name_to_display_name = dict(zip(name_df["name_in_notebook"].values, name_df["name"]))
name_to_order = dict(zip(name_df["name_in_notebook"].values, name_df["order"]))









# Input paths from a specific output directory from running the analysis pipeline.
input_paths = [
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_all_pmn_only_within_distances_melted.csv",
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_all_kegg_only_within_distances_melted.csv",
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_all_subsets_within_distances_melted.csv",
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_curated_pmn_only_within_distances_melted.csv",
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_curated_kegg_only_within_distances_melted.csv",
	"/Users/irbraun/phenologs-with-oats/outputs/stacked_11_13_2020_h15m38s01_8479_plants/stacked_curated_subsets_within_distances_melted.csv",
]









# Input paths from a specific output directory from running the analysis pipeline.
#plantcyc_pathways_cohesion_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_31_2020_h23m05s45_6169_plants/stacked_pmn_only_within_distances_melted.csv"
#phenotype_subsets_cohesion_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_31_2020_h23m05s45_6169_plants/stacked_subsets_within_distances_melted.csv"


# Output paths, figure out where to put these.
#output_path_plantcyc_pathways = "/Users/irbraun/phenologs-with-oats/outputs/plantcyc_pathways_cohesion_info.csv"
#output_path_phenotype_subsets = "/Users/irbraun/phenologs-with-oats/outputs/phenotype_subsets_cohesion_info.csv"


# Create and name an output directory according to when the notebooks was run and then create the paths for output files to put there.
#OUTPUT_NAME = "within"
#OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}".format(OUTPUT_NAME,datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
#os.mkdir(OUTPUT_DIR)
#output_path_plantcyc_pathways = os.path.join(OUTPUT_DIR,"plantcyc_pathways_cohesion_info.csv")
#output_path_phenotype_subsets = os.path.join(OUTPUT_DIR,"phenotype_subsets_cohesion_info.csv")





for input_path in input_paths:
	# Produce a table indicating how many of the biochemical pathways had significant cohesion values.
	# Working with the path.
	basename_without_extension = os.path.basename(input_path).split(".")[0]
	dirname = os.path.dirname(input_path)
	output_path = os.path.join(dirname, "{}_renamed.csv".format(basename_without_extension))
	# Working with the column subsets.
	df = pd.read_csv(input_path)
	df["name"] = df["approach"].map(name_to_display_name)
	df["order"] = df["approach"].map(name_to_order)
	df.drop_duplicates(subset=["order"], keep="first", inplace=True)
	df = df[["order","name","number_of_groups","fraction_significant"]]
	df["fraction_significant"] = df["fraction_significant"].map(lambda x:round(x,3))
	df.sort_values(by="order", inplace=True)
	df.to_csv(output_path, index=False)




print("done")