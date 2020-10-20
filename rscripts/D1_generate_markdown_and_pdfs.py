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








def create_markdown_and_pdfs_with_captions(dir_with_png_files, output_markdown_path, name_to_display_name, name_to_order)

	lines = []
	for filepath in glob.glob(os.path.join(dir_with_png_files,"*.png")):


		# For now, just extract all the information about captioning each image from the file names themselves.
		basename_without_extension = os.path.basename(filepath).replace(".png","").lower()


		# Looking in the filename for information about what to caption the image.
		# Note this is entirely dependedent on how the file is named when the image is created and saved.
		# This has to be consisent with what is done in the R script that generates and saves each plot.
		delim = "_curated_"
		approach_name = basename_without_extension.split(delim)[0]
		other_info = basename_without_extension.split(delim)[1]
		method = name_to_display_name.get(approach_name, "NAME NOT FOUND")
		order = name_to_order.get(approach_name, math.inf)
		if "true" in other_info:
			curation_info = "genes in the dataset that possessed annotations"
		else:
			curation_info = "all genes in the dataset"
		image_info = "The method used here is {}, applied to {}.".format(method, curation_info)

		lines.append((order, image_info, filepath))


	lines = sorted(lines, key=lambda x: x[0])

	with open(output_markdown_path, "w") as f:

		for (order, image_info, filepath) in lines:


			# Put whatever information is needed from above into a caption, and write the markdown line that links to that image.
			template = r"![{}]({})"
			line = template.format(image_info, filepath)
			f.write(line+"\n")
			f.write("\n")
			f.write("\n")
			f.write("\n")
			f.write(r"\pagebreak")
			f.write("\n")









# Read in the file that maps names used internally to names used in figures.
naming_dataframe_path = "/Users/irbraun/phenologs-with-oats/names.tsv"
name_df = pd.read_csv(naming_dataframe_path, sep="\t")
name_to_display_name = dict(zip(name_df["name_in_notebook"].values, name_df["name"]))
name_to_order = dict(zip(name_df["name_in_notebook"].values, name_df["order"]))




# Make the pairwise distribution files.
dir_with_png_files = "/Users/irbraun/phenologs-with-oats/figs/pairwise_distributions"
output_markdown_path = "/Users/irbraun/phenologs-with-oats/figs/distributions.md"
output_pdf_path = "/Users/irbraun/phenologs-with-oats/figs/distributions.pdf"
create_markdown_and_pdfs_with_captions(dir_with_png_files, output_markdown_path, name_to_display_name, name_to_order)
subprocess.run(["pandoc", output_markdown_path, "-o", output_pdf_path])


# Make the precision and recall curve files.
dir_with_png_files = "/Users/irbraun/phenologs-with-oats/figs/precision_recall_curves"
output_markdown_path = "/Users/irbraun/phenologs-with-oats/figs/pr_curves.md"
output_pdf_path = "/Users/irbraun/phenologs-with-oats/figs/pr_curves.pdf"
create_markdown_and_pdfs_with_captions(dir_with_png_files, output_markdown_path, name_to_display_name, name_to_order)
subprocess.run(["pandoc", output_markdown_path, "-o", output_pdf_path])


print("done")