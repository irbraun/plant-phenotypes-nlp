import sys
import os



# Add the path of the module with all functions needed.
PACKAGE_DIR = "/Users/irbraun/phenolog-db/src"
sys.path.append(PACKAGE_DIR)

import utils.nlp
import utils.similarity



# Packages that are either not on conda or the right versions aren't on conda.
# pip install pywsd==1.2.1
# pip install gensim==3.6.0
# pip install fastsemsim==1.0.0





d = {}
d["A"] = "the plants were super t,6^34all and the leaves were green"
d["B"] = "green plants a&^re ..usually very tall"
d["C"] = "other things can be green too"
d["D"] = "the dog walked"


d = {k:utils.nlp.get_clean_description(v) for (k,v) in d.items()}


print(utils.similarity.get_similarity_df_using_setofwords(d))
print(utils.similarity.get_similarity_df_using_bagofwords(d))












"""

Building the dicionary and other structures needed to create the pandas dataframes that hold the similarity values.

Read in IDs and corresponding chunks of text from some database or file.
clean texts and send them to be annotated to create an annotation corpus (ac) tsv file using the merged ontology.
clean texts and append related words for the bag-of-words and set-of-words strategies.
clean texts and don't append anything but add them to th
"""













