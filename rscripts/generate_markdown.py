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



dir_with_png_files = "/Users/irbraun/Desktop/test_folder_2"

output_markdown_path = "/Users/irbraun/phenologs-with-oats/figures/distributions.md"
output_pdf_path = "/Users/irbraun/phenologs-with-oats/figures/distributions.pdf"




with open(output_markdown_path, "w") as f:


	for filepath in glob.glob(os.path.join(dir_with_png_files,"*.png")):

		# For now, just extract all the information about captioning each image from the file names themselves.
		basename_without_extension = os.path.basename(filepath).replace(".png","").lower()
		caption = basename_without_extension
		image_info = basename_without_extension
		template = r"![{}]({})"
		line = template.format(image_info, filepath)
		f.write(line+"\n")
		f.write("\n")
		f.write("\n")
		f.write("\n")
		f.write(r"\pagebreak")
		f.write("\n")


subprocess.run(["pandoc", output_markdown_path, "-o", output_pdf_path])
print("done")