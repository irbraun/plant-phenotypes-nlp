import sys

sys.path.append("../.")
from phenolog.datasets.pathways import Pathways
from phenolog.utils.utils import save_to_pickle, load_from_pickle


# Create the pathways object.
species_dict = {
    "ath":"../data/pathways/plantcyc/aracyc_pathways.20180702", 
    "zma":"../data/pathways/plantcyc/corncyc_pathways.20180702", 
    "mtr":"../data/pathways/plantcyc/mtruncatulacyc_pathways.20180702", 
    "osa":"../data/pathways/plantcyc/oryzacyc_pathways.20180702", 
    "gmx":"../data/pathways/plantcyc/soycyc_pathways.20180702",
    "sly":"../data/pathways/plantcyc/tomatocyc_pathways.20180702"}


# Create and save the pathways object using PMN.
pathways = Pathways(species_dict, source="pmn")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/pmn_pathways.pickle")



# Create and save a pathways object using KEGG.
pathways = Pathways(species_dict, source="kegg")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/kegg_pathways.pickle")