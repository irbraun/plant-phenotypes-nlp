import sys
import os


# Adding the local modules that include all the methods used.
# This is actually relative to the dir where the script was called from, not where its located.
# ie . is ~ if the scripts was called as ./phenolog-db/src/notebooks/notebook2.py, figure out
# best way to deal with this.

sys.path.append("../.")
import utils.nlp
import utils.similarity
import utils.ontology
import utils.related


# Additional files needed to use ontologies and document embeddings.
merged_ontology_file = "../../ontologies/mo.obo"
annotations_file = "../../annotations/ac.tsv"
model_file = "../../gensim/apnews_dbow/doc2vec.bin"



# Create a sample dictionary of description IDs and descriptions.
d = {}
d["A"] = "the plants were super t,6^34all and the leaves were green"
d["B"] = "green plants a&^re ..usually very tall"
d["C"] = "other stuff can be green too"
d["D"] = "the dog walked and increased height etc leaves green"



td = utils.ontology.get_reverse_term_dictionary('/Users/irbraun/phenolog-db/ontologies/mo.obo')

a = utils.ontology.annotate_with_rabin_karp(d, td)
utils.ontology.write_annotations_to_tsv_file(a, "/Users/irbraun/Desktop/ac.tsv")