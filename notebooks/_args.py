"""
A file to store big lists of calls to the Method constructor that can be copied and pasted into the notebooks.
"""








training_methods = [

	
	# Full phenotype descriptions


	# Methods that use neural networks to generate embeddings.
	Method("Doc2Vec", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
	Method("Doc2Vec", "PubMed,Size=100", pw.pairwise_square_doc2vec, {"model":doc2vec_pubmed_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Mean,Nouns,Adjectives", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions_noun_adj_simple_preprocessing, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Max,Nouns,Adjectives", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions_noun_adj_simple_preprocessing, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Mean,Linares Pontes", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
	Method("Word2Vec", "Wikipedia,Size=300,Max,Linares Pontes", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
	
	Method("BERT", "Base:Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
	Method("BERT", "Base:Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
	Method("BERT", "Base:Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
	Method("BERT", "Base:Layers=2,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine),
	Method("BERT", " Base:Layers=3,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine),
	Method("BERT", "Base:Layers=4,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine),
	Method("BioBERT", "PMC,Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
	Method("BioBERT", "PMC,Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
	Method("BioBERT", "PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
	Method("BioBERT", "PubMed,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
	Method("BioBERT", "PubMed,PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),


	# Methods that use topic models to generate embeddings.
	Method("Topic Models", "LDA,Simple,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":20, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Simple,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":50, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Simple,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":200, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":20, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":50, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"lda"}, spatial.distance.cosine),
	Method("Topic Models", "LDA,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":200, "algorithm":"lda"}, spatial.distance.cosine),

	Method("Topic Models", "NMF,Simple,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":20, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":50, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "num_topics":200, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":20, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":50, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "num_topics":200, "algorithm":"nmf"}, spatial.distance.cosine),

	Method("Topic Models", "NMF,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
	Method("Topic Models", "NMF,Simple,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),


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
	Method("N-Grams", "Full,Nouns,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
	Method("N-Grams", "Full,Nouns,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
	Method("N-Grams", "Full,Nouns,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
	Method("N-Grams", "Full,Nouns,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
	
	# Methods that use variations on the n-grams approach with a reduced vocabulary size and simple preprocessing (no stemming).
	Method("N-Grams", "Simple,Words,Linares Pontes,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
	Method("N-Grams", "Simple,Words,Linares Pontes,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
	Method("N-Grams", "Simple,Words,Linares Pontes,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
	Method("N-Grams", "Simple,Words,Linares Pontes,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),



	# Methods that use terms inferred from automated annotation of the text.
	Method("NOBLE Coder", "Precise", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
	Method("NOBLE Coder", "Partial", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
	Method("NOBLE Coder", "Precise,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
	Method("NOBLE Coder", "Partial,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
	
	# Methods that use terms assigned by humans that are present in the dataset.
	Method("GO", "None", pw.pairwise_square_annotations, {"ids_to_annotations":go_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
	Method("PO", "None", pw.pairwise_square_annotations, {"ids_to_annotations":po_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),

	
  



	# Individual phenes from those larger phenotypes.


	# Approaches were the phenotype descriptions were split into individual phenes first (computationally expensive).
	Method("Doc2Vec (Phenes)", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":phenes, "metric":"cosine"}, spatial.distance.cosine, tag="phenes"),
	Method("Doc2Vec (Phenes)", "PubMed,Size=100", pw.pairwise_square_doc2vec, {"model":doc2vec_pubmed_model, "ids_to_texts":phenes, "metric":"cosine"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"mean"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"max"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Mean,Nouns,Adjectives", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes_noun_adj_simple_preprocessing, "metric":"cosine", "method":"mean"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Max,Nouns,Adjectives", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes_noun_adj_simple_preprocessing, "metric":"cosine", "method":"max"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Mean,Linares Pontes", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes_linares_pontes, "metric":"cosine", "method":"mean"}, spatial.distance.cosine, tag="phenes"),
	Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Max,Linares Pontes", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes_linares_pontes, "metric":"cosine", "method":"max"}, spatial.distance.cosine, tag="phenes"),

	Method("BERT (Phenes)", "Base:Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine, tag="phenes"),
	Method("BERT (Phenes)", "Base:Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine, tag="phenes"),
	Method("BERT (Phenes)", "Base:Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
	Method("BERT (Phenes)", "Base:Layers=2,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine, tag="phenes"),
	Method("BERT (Phenes)", "Base:Layers=3,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine, tag="phenes"),
	Method("BERT (Phenes)", "Base:Layers=4,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine, tag="phenes"),
	Method("BioBERT (Phenes)", "PMC,Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine, tag="phenes"),
	Method("BioBERT (Phenes)", "PMC,Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine, tag="phenes"),
	Method("BioBERT (Phenes)", "PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
	Method("BioBERT (Phenes)", "PubMed,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
	Method("BioBERT (Phenes)", "PubMed,PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),


	
	# Methods that use variations on the N-Grams (Phenes) approach with full preprocessing (includes stemming).
	Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	
	# Methods that use variations on the N-Grams (Phenes) approach with simple preprocessing (no stemming).
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	
	# Methods that use variations on the N-Grams (Phenes) approach selecting for specific parts-of-speech (includes stemming).
	Method("N-Grams (Phenes)", "Full,Nouns,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Full,Nouns,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	
	# Methods that use variations on the N-Grams (Phenes) approach with a reduced vocabulary size and simple preprocessing (no stemming).
	Method("N-Grams (Phenes)", "Simple,Words,Linares Pontes,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,Linares Pontes,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,Linares Pontes,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
	Method("N-Grams (Phenes)", "Simple,Words,Linares Pontes,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),



]











