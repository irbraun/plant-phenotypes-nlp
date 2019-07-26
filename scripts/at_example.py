import sys
import os
import pandas as pd


sys.path.append("../.")
import phenolog.nlp
import phenolog.similarity
import phenolog.ontology
import phenolog.related




# Read in a dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/Locus_Germplasm_Phenotype_20180702.txt"
usecols = ["LOCUS_NAME", "GERMPLASM_NAME", "PHENOTYPE", "PUBMED_ID"]
names = ["locus", "germplasm", "phenotype", "pubmed_id"]
renamed = {k:v for k,v in zip(usecols,names)}
df = pd.read_table(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["id"] = [str(i) for i in df.index.values]


# Use a small subset of the data for this example.
df = df.head(100)



# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = {identifier:description for (identifier,description) in zip(df.id,df.phenotype)}
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}




# Prepare a version of the dictionary where descriptions are expanded to include related words.
word2vec_model_file = "../gensim/wiki_sg/word2vec.bin"
word2vec_model = phenolog.nlp.load_word2vec_model(word2vec_model_file)
description_dict_expanded = {i:phenolog.nlp.append_related_words(d, phenolog.related.get_all_word2vec_related_words(d,word2vec_model,0.5,10)) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/at_annotations.tsv"
doc2vec_model_file = "../gensim/apnews_dbow/doc2vec.bin"

term_dict = phenolog.ontology.get_reverse_term_dictionary(ontology_obo_file=merged_ontology_file)
annotations = phenolog.ontology.annotate_with_rabin_karp(object_dict=description_dict, term_dict=term_dict)
phenolog.ontology.write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
print(phenolog.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict).head())
print(phenolog.similarity.get_similarity_df_using_doc2vec(doc2vec_model_file, description_dict).head())
print(phenolog.similarity.get_similarity_df_using_bagofwords(description_dict).head())
print(phenolog.similarity.get_similarity_df_using_setofwords(description_dict).head())