#!/usr/bin/env python
# coding: utf-8

# In[228]:


import pandas as pd
import numpy as np
import gensim
import random
from nltk import sent_tokenize
from nltk import word_tokenize
import warnings
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

warnings.simplefilter('ignore')


# In[229]:


# Input paths to text datasets.
plant_abstracts_corpus_path = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt"
plant_phenotype_descriptions_path = "../../plant-data/genes_texts_annots.csv"


# In[233]:


# Preparing the dataset that combines the dataset of plant phenotype descriptions and scrapped abstracts.
corpus = open(plant_abstracts_corpus_path, 'r').read()
sentences_from_corpus = sent_tokenize(corpus)
phenotype_descriptions = " ".join(pd.read_csv(plant_phenotype_descriptions_path)["descriptions"].values)
times_to_duplicate_phenotype_dataset = 5
sentences_from_descriptions = sent_tokenize(phenotype_descriptions)
sentences_from_descriptions = list(np.repeat(sentences_from_descriptions, times_to_duplicate_phenotype_dataset))
sentences = sentences_from_corpus+sentences_from_descriptions
random.shuffle(sentences)
sentences = [preprocess_string(sentence) for sentence in sentences]
print(len(sentences))


# In[ ]:


print("starting training")


# In[150]:


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epochs = []
        self.epoch = 1
        self.losses = []
        self.deltas = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            delta = loss
        else:
            delta = loss- self.loss_previous_step
        self.loss_previous_step=loss
        self.losses.append(loss)
        self.epochs.append(self.epoch)
        self.epoch += 1
        self.deltas.append(delta)


# In[235]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=50, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim50.model"
model.save(output_path)
print("done training 1 ")


# In[ ]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=100, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim100.model"
model.save(output_path)
print("done training 2")


# In[ ]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=150, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim150.model"
model.save(output_path)
print("done training 3")


# In[ ]:


print(stophere)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[199]:


preprocess_string("plants leaves genes proteins tall wide abscisic root nodule")


# In[200]:


from scipy.spatial.distance import cosine
print(cosine(model[["plant"]],model[["leav"]]))
print(cosine(model[["gene"]],model[["leav"]]))
print(cosine(model[["gene"]],model[["protein"]]))
print(cosine(model[["auxin"]],model[["hormon"]]))
print(cosine(model[["hormon"]],model[["abscis"]]))
print(cosine(model[["root"]],model[["nodul"]]))
len(model.wv.vocab)


# In[201]:


# Checking to make sure the model can be loaded and used for looking up embeddings.
path = "../models/wiki_sg/word2vec.bin"
model_from_wikipedia = gensim.models.Word2Vec.load(path)
a_word_in_vocab = list(model_from_wikipedia.wv.vocab.keys())[0]
vector = model_from_wikipedia[a_word_in_vocab]
print(len(vector))


# In[221]:


len(model.wv.vocab)


# In[206]:


model.intersect_word2vec_format("../models/wiki_sg/word2vec.bin", binary=True, lockf=1.0)
model.train(sentences, epochs=1, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])
len(model.wv.vocab)


# In[ ]:





# In[ ]:




