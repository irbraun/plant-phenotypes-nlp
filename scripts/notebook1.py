import sys
import os


# Adding the local modules that include all the methods used.
sys.path.append("../.")
import utils.nlp
import utils.similarity
import utils.onto






# Additional files needed to use ontologies and document embeddings.
merged_ontology_file = "../../ontologies/mo.obo"
annotations_file = "../../annotations/ac.tsv"
model_file = "../../gensim/apnews_dbow/doc2vec.bin"





# Create a sample dictionary of description IDs and descriptions.
d = {}
d["A"] = "the plants were super t,6^34all and the leaves were green"
d["B"] = "green plants a&^re ..usually very tall"
d["C"] = "other things can be green too"
d["D"] = "the dog walked"



# Clean the text descriptions.
d = {k:utils.nlp.get_clean_description(v) for (k,v) in d.items()}




# Generate the dataframes of similarity values using each method.
print(utils.similarity.get_similarity_df_using_setofwords(d))
print(utils.similarity.get_similarity_df_using_bagofwords(d))
print(utils.similarity.get_similarity_df_using_doc2vec(model_file, d))
print(utils.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, d))



#from Bio.Blast import NCBIWWW
#help(NCBIWWW.qblast)
#seq = "MEVQLGLGRVYPRPPSKTYRGAFQNLFQSVREVIQNPGPRHPEAAAAAAAAAAAAASAAPPGAHLQQQQETSPRQQQQQGEDGSPQTQSRGPTGYLALAREAAGAPTCSKDSYLGCSSTIS"
#result = NCBIWWW.qblast(program="blastp", database="refseq_genomic", entrez_query="txid6656[ORGN]", sequence=seq, hitlist_size=2) 