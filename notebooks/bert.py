#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import BertModel, BertForMaskedLM, BertConfig
from torch.utils.data import TensorDataset, random_split
from ipywidgets import IntProgress
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, AdamW

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
from sklearn.metrics import precision_recall_curve
from itertools import product


# In[40]:


# If there's a GPU available...
using_gpu = False
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    using_gpu = True
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


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


# In[4]:


# Preprocess the sentences to be stemmed tokens separated by whitespace and show the first few of them.
sentences = [" ".join(s) for s in sentences_from_corpus_and_descriptions]
sentences = [:50]
sentences[:5]


# In[5]:


# Preparing a vocabulary file based on these sentences for BERT.
vocabulary_file_path = "../data/corpus_related_files/vocabulary/vocab.txt"
vocabulary = set()
for s in sentences_from_corpus_and_descriptions:
    vocabulary.update(s)
vocabulary.update(["[PAD]","[SEP]","[UNK]","[MASK]"])
print("there are now {} words in the vocabulary including special tokens".format(len(vocabulary)))
vocabulary_size = len(vocabulary)
print("the first few are")
print(list(vocabulary)[:8])
with open(vocabulary_file_path, "w") as f:
    for token in list(vocabulary):
        f.write(token+"\n")
print("done writing to the vocabulary file")


# In[41]:


# Creating and parameratizing the small BERT architecture.
vocab_size = vocabulary_size
small_bert_configuration = BertConfig(
    vocab_size=vocab_size, 
    hidden_size=50, 
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=100,
    max_position_embeddings=200,
    return_dict=True,   
)
model = BertForMaskedLM(small_bert_configuration)
if using_gpu:
    model.cuda()

# An easier to read description of the model, from BERT fine-tuning with PyTorch.
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# In[7]:


# Creating the tokenizer using the provided vocabulary.
tokenizer = BertTokenizer(vocab_file=vocabulary_file_path)
print(tokenizer)


# In[8]:


# Testing out the tokenizer with the first few sentences from the dataset.
# Note that all three arrays for all the sentences should be of the same length.
# The attention mask should indicate whether a position is padding token or not, and token type IDs are not used.
# The input IDs refer to the vocabulary ID of that particular word.
encoding = tokenizer(sentences[0:3], return_tensors='pt', padding=True, truncation=True)
print(encoding.input_ids)
print(encoding.token_type_ids)
print(encoding.attention_mask)
print(encoding.input_ids.shape)
print(encoding.token_type_ids.shape)
print(encoding.attention_mask.shape)


# In[9]:


# Producing the corresponding dataset of sentences with masked tokens for training.
# Probability of a token getting masked should be set here.
# Show the first few entries in the masked dataset to verify that some tokens are swapped with [MASK].
prob = 0.15
masked = [" ".join([np.random.choice(['[MASK]',token],p=[prob,1-prob]) for token in s.split()]) for s in sentences]
masked[:5]


# In[10]:


# Preparing the dataset object that can be read in as batches during the training loop.
inputs_dict = tokenizer(masked, return_tensors='pt', padding=True, truncation=True)
labels_dict = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
print(inputs_dict["input_ids"].shape)
print(labels_dict["input_ids"].shape)
dataset = TensorDataset(inputs_dict["input_ids"], inputs_dict["attention_mask"], labels_dict["input_ids"])

# Pick the batch size here.
batch_size = 32
train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)


# In[35]:


# Creating and parameterizing the necessary objects for the optimizer and learning rate scheduler.
epochs = 5
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
total_steps = len(train_dataloader)*epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_steps)
print(total_steps)


# In[32]:


# Setting up a validation task to monitor to avoid overfitting to the language modeling task.
validation_df = pd.read_csv("../data/corpus_related_files/closely_related/concepts_multi_word.csv")
concepts_1 = list(validation_df["concept_1"].values)
concepts_2 = list(validation_df["concept_2"].values)
random.shuffle(concepts_2)
validation_df_shuffled = pd.DataFrame({"concept_1":concepts_1,"concept_2":concepts_2})
validation_df["class"] = 1
validation_df_shuffled["class"] = 0
df = pd.concat([validation_df,validation_df_shuffled])
output_path_for_results = "../models/bert_small/{}_validation.csv".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
output_path_for_results_summary = "../models/bert_small/{}_validation_summary.csv".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
rows = []


# In[33]:


# Setting up function to get sentence embeddings out of the word model.
def vectorize_with_bert(text, model, tokenizer, method="sum", layers=4):

    #This function uses a pretrained BERT model to infer a document level vector for a collection 
    #of one or more sentences. The sentence are defined using the nltk sentence parser. This is 
    #done because the BERT encoder expects either a single sentence or a pair of sentences. The
    #internal representations are drawn from the last n layers as specified by the layers argument, 
    #and represent a particular token but account for the context that it is in because the entire
    #sentence is input simultanously. The vectors for the layers can concatentated or summed 
    #together based on the method argument. The vector obtained for each token then are averaged
    #together to for the document level vector.
    # This is just copied and pasted from the oats function.
    #Args:
        #text (str):  Any arbitrary text string.
        #model (pytorch model): An already loaded BERT PyTorch model from a file or other source.
        #tokenizer (bert tokenizer): Object which handles how tokenization specific to BERT is done. 
        #method (str): A string indicating how layers for a token should be combined (concat or sum).
        #layers (int): An integer saying how many layers should be used for each token.

    sentences = sent_tokenize(text)
    token_vecs_cat = []
    token_vecs_sum = []

    for text in sentences:
        marked_text = "{} {} {}".format("[CLS]",text,"[SEP]")
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        with torch.no_grad():
            #encoded_layers,_ = model(tokens_tensor,segments_tensor)
            
            
            # Because the model configuration is a little bit different, the forward pass call is modified.
            outputs = model(tokens_tensor,segments_tensor, output_hidden_states=True)
            encoded_layers = outputs.hidden_states
            
            
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = token_embeddings.permute(1,2,0,3)
        batch = 0
        for token in token_embeddings[batch]:
            concatenated_layer_vectors = torch.cat(tuple(token[-layers:]), dim=0)
            summed_layer_vectors = torch.sum(token[-layers:], dim=0)
            token_vecs_cat.append(np.array(concatenated_layer_vectors))
            token_vecs_sum.append(np.array(summed_layer_vectors))

    # Check to make sure atleast one token was found with an embedding to use as a the 
    # vector representation. If there wasn't found, this is because of the combination
    # of what the passed in description was, and how it was handled by either the sentence
    # tokenizing step or the BERT tokenizer methods. Handle this by generating a random
    # vector. This makes the embedding meaningless but prevents multiple instances that
    # do not have embeddings from clustering together in downstream analysis. An expected
    # layer size is hardcoded for this section based on the BERT architecture.
    expected_layer_size = 50
    if len(token_vecs_cat) == 0:
        print("no embeddings found for input text '{}', generating random vector".format(description))
        random_concat_vector = np.random.rand(expected_layer_size*layers)
        random_summed_vector = np.random.rand(expected_layer_size)
        token_vecs_cat.append(random_concat_vector)
        token_vecs_sum.append(random_summed_vector)

    # Average the vectors obtained for each token across all the sentences present in the input text.
    if method == "concat":
        embedding = np.mean(np.array(token_vecs_cat),axis=0)
    elif method == "sum":
        embedding = np.mean(np.array(token_vecs_sum),axis=0)
    else:
        raise ValueError("method argument is invalid")
    return(embedding)


# In[34]:


# The training loop that uses batches from that data loader.
for epoch_i in range(0, epochs):
    model.train()
    total_train_loss = 0 
    for step,batch in enumerate(train_dataloader):    
        model.zero_grad()
        
        if using_gpu:
            bi = batch[0].to(device)
            bm = batch[1].to(device)
            bl = batch[2].to(device)
        else:
            bi = batch[0]
            bm = batch[1]
            bl = batch[2]
        
        
        
        outputs = model(input_ids=bi, attention_mask=bm, labels=bl)
        loss = outputs.loss
        logits = outputs.logits
        print(step, loss)
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader) 
    print("{} {}".format(epoch_i, avg_train_loss))
    
    
    # Keeping track of performance on the simple validation task by getting a precision recall curve.
    model_name = "bert_small"
    vectorize = lambda x: vectorize_with_bert(" ".join(preprocess_string(x)), model, tokenizer, "sum", 1)
    get_similarity = lambda s1,s2: 1-cosine(vectorize(s1),vectorize(s2))
    df[model_name] = df.apply(lambda x: get_similarity(x["concept_1"],x["concept_2"]),axis=1)
    y_true = list(df["class"].values)
    y_prob = list(df[model_name].values)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
    f_1_scores = f_beta(precision,recall,beta=1)
    f_1_max = np.nanmax(f_1_scores)
    rows.append((model_name, epoch_i, loss.item(), f_1_max))
          


# Writing results of the validation to those files.
df.to_csv(output_path_for_results, index=False)
header = ["model","epoch","training_loss","f1_max"]
pd.DataFrame(rows, columns=header).to_csv(output_path_for_results_summary, index=False)


# In[44]:


output_dir = "../models/bert_small/model_save_{}/".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model 
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

