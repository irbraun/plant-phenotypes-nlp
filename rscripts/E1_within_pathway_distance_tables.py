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
plantcyc_pathways_cohesion_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_31_2020_h23m05s45_6169_plants/stacked_pmn_only_within_distances_melted.csv"
phenotype_subsets_cohesion_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_31_2020_h23m05s45_6169_plants/stacked_subsets_within_distances_melted.csv"


# Output paths, figure out where to put these.
output_path_plantcyc_pathways = "/Users/irbraun/phenologs-with-oats/outputs/plantcyc_pathways_cohesion_info.csv"
output_path_phenotype_subsets = "/Users/irbraun/phenologs-with-oats/outputs/phenotype_subsets_cohesion_info.csv"


# Create and name an output directory according to when the notebooks was run and then create the paths for output files to put there.
OUTPUT_NAME = "within"
OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}".format(OUTPUT_NAME,datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
os.mkdir(OUTPUT_DIR)
output_path_plantcyc_pathways = os.path.join(OUTPUT_DIR,"plantcyc_pathways_cohesion_info.csv")
output_path_phenotype_subsets = os.path.join(OUTPUT_DIR,"phenotype_subsets_cohesion_info.csv")









# Produce a table indicating how many of the biochemical pathways had significant cohesion values.
df = pd.read_csv(plantcyc_pathways_cohesion_path)
df["name"] = df["approach"].map(name_to_display_name)
df["order"] = df["approach"].map(name_to_order)
df.drop_duplicates(subset=["order"], keep="first", inplace=True)
df = df[["order","name","total_groups","num_significant","num_adjusted","fraction_significant","fraction_adjusted"]]
df["fraction_adjusted"] = df["fraction_adjusted"].map(lambda x:round(x,3))
df["fraction_significant"] = df["fraction_significant"].map(lambda x:round(x,3))
df.sort_values(by="order", inplace=True)
df.to_csv(output_path_plantcyc_pathways, index=False)



# Produce a table indicating how many of the phenotype subsets had significant cohesion values.
df = pd.read_csv(phenotype_subsets_cohesion_path)
df["name"] = df["approach"].map(name_to_display_name)
df["order"] = df["approach"].map(name_to_order)
df.drop_duplicates(subset=["order"], keep="first", inplace=True)
df = df[["order","name","total_groups","num_significant","num_adjusted","fraction_significant","fraction_adjusted"]]
df["fraction_adjusted"] = df["fraction_adjusted"].map(lambda x:round(x,3))
df["fraction_significant"] = df["fraction_significant"].map(lambda x:round(x,3))
df.sort_values(by="order", inplace=True)
df.to_csv(output_path_phenotype_subsets, index=False)

print("done")