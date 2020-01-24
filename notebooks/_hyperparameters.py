"""
A file to store big lists of calls to the Method constructor that can be copied and pasted into the notebooks.
"""








methods = [

    
    # Methods that use neural networks to generate embeddings.
    Method("Doc2Vec Wikipedia", "Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    Method("Doc2Vec PubMed", "Size=100", pw.pairwise_square_doc2vec, {"model":doc2vec_pubmed_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    Method("Word2Vec Wikipedia", "Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
    Method("Word2Vec Wikipedia", "Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
    
    Method("BERT", "Base:Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    Method("BERT", " Base:Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    Method("BERT", " Base:Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    Method("BERT", " Base:Layers=2,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine),
    Method("BERT", " Base:Layers=3,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine),
    Method("BERT", " Base:Layers=4,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine),
    Method("BioBERT", "PMC,Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    Method("BioBERT", "PMC,Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    Method("BioBERT", "PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    Method("BioBERT", "PubMed,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    Method("BioBERT", "PubMed,PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
        
    # Methods that use variations on the n-grams approach with full preprocessing (includes stemming).
    Method("N-Grams", "Full,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    
    # Methods that use variations on the n-grams approach with simple preprocessing (no stemming).
    Method("N-Grams", "Simple,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Simple,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    
    # Methods that use variations on the n-grams approach selecting for specific parts-of-speech (includes stemming).
    Method("N-Grams", "Full,Nouns,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Nouns,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Nouns,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Nouns,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    
    # Methods that use variations on the n-grams approach with a reduced vocabulary size and simple preproocessing (no stemming).
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),



    # Methods that use terms inferred from automated annotation of the text.
    Method("NOBLE Coder", "Precise", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Partial", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Precise,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
    Method("NOBLE Coder", "Partial,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
    
    # Methods that use terms assigned by humans that are present in the dataset.
    Method("GO", "Default", pw.pairwise_square_annotations, {"ids_to_annotations":go_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
    Method("PO", "Default", pw.pairwise_square_annotations, {"ids_to_annotations":po_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),


]