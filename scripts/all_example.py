import sys
import os
import pandas as pd


sys.path.append("../.")
from phenolog.Dataset import Dataset
import phenolog.nlp
import phenolog.similarity
import phenolog.ontology
#import phenolog.related
import phenolog.combine





# Create a dataset object that can be added to.
dataset = Dataset()



# Read in the phenotype descriptions dataframe from the Oellrich, Walls et al dataset.
filename = "/Users/irbraun/phenolog/data/ppn/oellrich_walls_textdata_only.csv"
usecols = ["species", "locus", "phenotype"]
usenames = ["species", "locus", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)
df["pubmed"] = "unknown"
dataset.add_data(df)


print(pd.unique(df.species))
sys.exit()




# Read in the phene descriptions dataframe from the Oellrich, Walls et al dataset.
filename = "/Users/irbraun/phenolog/data/ppn/oellrich_walls_textdata_only.csv"
usecols = ["species", "locus", "phene"]
usenames = ["species", "locus", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)
df["pubmed"] = "unknown"
dataset.add_data(df)



# Read in a dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/tair/Locus_Germplasm_Phenotype_20180702.txt"
usecols = ["LOCUS_NAME", "GERMPLASM_NAME", "PHENOTYPE", "PUBMED_ID"]
usenames = ["locus", "germplasm", "description", "pubmed"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_table(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["species"] = "ath"
dataset.add_data(df)




# Read in the dataframe from MaizeGDB table.





# See what the full dataset contains after adding the individual files.
dataset.check()
dataset.subsample_data(num_to_retain=5)
dataset.check()

# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary(all_rows=1)
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.

merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/all_annotations.tsv"
doc2vec_model_file = "../gensim/apnews_dbow/doc2vec.bin"

term_dicts = phenolog.ontology.get_term_dictionaries(ontology_obo_file=merged_ontology_file)
term_dict_fwd = term_dicts[0]
term_dict_rev = term_dicts[1]
annotations = phenolog.ontology.annotate_with_rabin_karp(object_dict=description_dict, term_dict=term_dict_rev)
phenolog.ontology.write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
df1 = phenolog.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict)
df2 = phenolog.similarity.get_similarity_df_using_doc2vec(doc2vec_model_file, description_dict)
df3 = phenolog.similarity.get_similarity_df_using_bagofwords(description_dict)
df4 = phenolog.similarity.get_similarity_df_using_setofwords(description_dict)

print(df1)
print(df2)
print(df3)
print(df4)

dfs = [df1, df2, df3, df4]
df_names = ["a","b","c","d"]
weights = [0.014, 0.914, 0.004, 0.006]

df_dict = {k:v for (k,v) in zip(df_names,dfs)}
wt_dict = {k:v for (k,v) in zip(df_names,weights)}


target_df = df4


reg_model = phenolog.combine.learn_weights_linear_regression(df_dict, target_df)
df = phenolog.combine.combine_with_linear_model(df_dict, reg_model)
print(df)

#df = phenolog.combine.combine_with_weights(df_dict, wt_dict)
#print("the combined one")
#print(df)











