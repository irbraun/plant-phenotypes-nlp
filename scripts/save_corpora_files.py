from nltk.corpus import brown
import nltk
import sys
nltk.download('punkt')
nltk.download('brown')

sys.path.append("../../oats")
from oats.nlp.vocabulary import vocabulary_by_feature_selection
from oats.pubmed.querying import search, fetch_details





# Where to save the corpora files from either background or interesting text.
background_path = "../data/corpora_related_files/background.txt"
ofinterest_path = "../data/corpora_related_files/phenotypes.txt"

# Getting a sampling of text that could be considered background or general information.
background_text = " ".join(brown.words(categories=['news',"editorial","reviews","lore"]))
background_file = open(background_path,"w")
background_file.write(background_text)
background_file.close()

# Getting a sampling of text that is from some domain of interest.
interesting_texts = []
limit = 1000
query = "arabidopsis AND phenotype"
results = search(query, limit)
id_list = results['IdList']
if len(id_list) > 0:
    papers = fetch_details(id_list)
    for i, paper in enumerate(papers['PubmedArticle']):
        try:
            abstract_text = paper['MedlineCitation']['Article']['Abstract']["AbstractText"][0]
            interesting_texts.append(abstract_text)
        except KeyError:
            continue
interesting_text = " ".join(interesting_texts)
ofinterest_file = open(ofinterest_path,"w")
ofinterest_file.write(interesting_text)
ofinterest_file.close()