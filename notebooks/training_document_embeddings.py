#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gensim
import random
import sys
import glob
import os
import datetime
from nltk import sent_tokenize
from nltk import word_tokenize
from scipy.spatial.distance import cosine
import warnings
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from itertools import product
from sklearn.metrics import precision_recall_curve

warnings.simplefilter('ignore')

sys.path.append("../../oats")
from oats.annotation.ontology import Ontology
from oats.distances import pairwise as pw
from oats.utils.utils import flatten


# ### 1. Creating datasets of sentences to traing word embedding models

# In[2]:


# Input paths to text datasets.
plant_abstracts_corpus_path = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt"
plant_phenotype_descriptions_path = "../../plant-data/genes_texts_annots.csv"


# In[3]:


# Preparing the dataset that combines the dataset of plant phenotype descriptions and scrapped abstracts.
corpus = open(plant_abstracts_corpus_path, 'r').read()
sentences_from_corpus = sent_tokenize(corpus)
phenotype_descriptions = " ".join(pd.read_csv(plant_phenotype_descriptions_path)["descriptions"].values)
times_to_duplicate_phenotype_dataset = 5
sentences_from_descriptions = sent_tokenize(phenotype_descriptions)
sentences_from_descriptions_duplicated = list(np.repeat(sentences_from_descriptions, times_to_duplicate_phenotype_dataset))
sentences_from_corpus_and_descriptions = sentences_from_corpus+sentences_from_descriptions_duplicated
random.shuffle(sentences_from_corpus_and_descriptions)
random.shuffle(sentences_from_corpus)
random.shuffle(sentences_from_descriptions)
sentences_from_corpus_and_descriptions = [preprocess_string(s) for s in sentences_from_corpus_and_descriptions]
sentences_from_corpus = [preprocess_string(s) for s in sentences_from_corpus]
sentences_from_descriptions = [preprocess_string(s) for s in sentences_from_descriptions]
assert len(sentences_from_corpus_and_descriptions) == len(sentences_from_corpus)+(times_to_duplicate_phenotype_dataset*len(sentences_from_descriptions))
print(len(sentences_from_corpus_and_descriptions))
print(len(sentences_from_corpus))
print(len(sentences_from_descriptions))


# ### 2. Training and saving models with hyperparameter grid search

# In[50]:


# Defining grid search parameters for training word embedding models.
training_sentence_sets =  [(sentences_from_corpus_and_descriptions,"both"), (sentences_from_corpus,"abstracts"), (sentences_from_descriptions,"dataset")]
training_sentence_sets =  [(sentences_from_corpus_and_descriptions,"both")]
dimensions = [(x,"dim{}".format(str(x))) for x in [200, 300]]
num_epochs = [(x,"ep{}".format(str(x))) for x in [1]]
min_alpha = [(0.0001,"a")]
alpha = [(0.025,"s")]
min_count = [(x,"min{}".format(str(x))) for x in [3]]
hyperparameter_sets = list(product(
    num_epochs,
    training_sentence_sets, 
    dimensions, 
    min_alpha, 
    alpha, 
    min_count))
print(len(hyperparameter_sets))


# In[ ]:


def train_and_save_document_embedding_model(output_dir, hyperparameters):
    
    # Producing the path to the output file named according to hyperparameters.
    model_filename = "doc2vec_{}.model".format("_".join([x[1] for x in hyperparameters]))
    output_path = os.path.join(output_dir, model_filename)    
    
    # Get the hyperparamter values.
    epochs, sentences, vector_size, min_a, a, min_count = [x[0] for x in hyperparameters]  
    print(epochs, sentences[0], vector_size, min_a, a, min_cont)
    
    # Fitting a vocabulary and training the model.
    tagged_sentences = [TaggedDocument(words=s,tags=[i]) for i,s in enumerate(sentences)]
    workers = 4
    model = gensim.models.Doc2Vec(vector_size=vector_size, min_count=min_count, dm=0, workers=workers, alpha=a, min_alpha=min_a, dbow_words=0)
    model.build_vocab(tagged_sentences)
    model.train(tagged_sentences, epochs=epochs, total_examples=model.corpus_count)   
    
    # Saving the model to a file.
    model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
    model.save(output_path)
    print("saving {}".format(model_filename))


# In[ ]:


# Calling the model creation function iteratively through the hyperparameter grid search.
output_models_directory = "../models/plants_dbow"
for h in hyperparameter_sets:
    train_and_save_document_embedding_model(output_models_directory, h)


# ### 3. Evaluating word embedding models on the validation dataset of related concepts (sentence version)

# In[96]:


models_dir = "../models/plants_dbow"

output_path_for_results = "../models/plants_dbow/{}_validation.csv".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
output_path_for_results_summary = "../models/plants_dbow/{}_validation_summary.csv".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))

rows = []
     
validation_df = pd.read_csv("../data/corpus_related_files/closely_related/concepts_multi_word.csv")
concepts_1 = list(validation_df["concept_1"].values)
concepts_2 = list(validation_df["concept_2"].values)
random.shuffle(concepts_2)
validation_df_shuffled = pd.DataFrame({"concept_1":concepts_1,"concept_2":concepts_2})
validation_df["class"] = 1
validation_df_shuffled["class"] = 0





df = pd.concat([validation_df,validation_df_shuffled])


for path in glob.glob(os.path.join(models_dir,"*.model")): 
    model = gensim.models.Doc2Vec.load(path)
    model_name = os.path.basename(path)    
    get_similarity = lambda s1,s2: 1-cosine(model.infer_vector(preprocess_string(s1)),model.infer_vector(preprocess_string(s2)))
    df[model_name] = df.apply(lambda x: get_similarity(x["concept_1"],x["concept_2"]),axis=1)
    
    
    y_true = list(df["class"].values)
    y_prob = list(df[model_name].values)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
    f_1_scores = f_beta(precision,recall,beta=1)
    f_1_max = np.nanmax(f_1_scores)
          
    rows.append((model_name, f_1_max))

    
    
          
# Constructing and saving the results dataframe.        
df.to_csv(output_path_for_results, index=False)
header = ["model","f1_max"]
pd.DataFrame(rows, columns=header).to_csv(output_path_for_results_summary, index=False)
df

