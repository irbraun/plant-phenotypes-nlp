import sys
import os
import pandas as pd
import numpy as np


sys.path.append("../.")
from phenolog.annotation.ontology import Ontology
from phenolog.annotation import batch
from phenolog.annotation import annotation




merged_ontology_file = "../ontologies/mo.obo"
onto = Ontology(merged_ontology_file)



for term in onto.pronto_ontology_obj:
	print("{} \t\t {}".format(onto.ic_dict[term.id], term.name))

for k,v in onto.reverse_term_dict.items():
	print("{} {}".format(k,v))

for k,v in onto.forward_term_dict.items():
	print("{} {}".format(k,v))


'''
for term in o.onto_obj:
	term_list.append(term.id)
	name_list.append(term.name)






id1 = term_list[40]
n = name_list[40]
print(id1)
print(n)
id2 = term_list[135]
n = name_list[135]
print(id2)
print(n)

s = o.get_unweighted_jaccard_similarity_of_terms(id1, id2)

print(s)

'''
