from Bio.KEGG import REST
from collections import defaultdict
import pandas as pd
import numpy as np
import re
import itertools


import phenolog.utils





class Pathways:




    def __init__(self, species_dict, source):


        # Create one dataframe and pair of dictionaries for each species code.
        self.species_list = species_dict.keys()
        self.species_to_df_dict = {}
        self.species_to_fwd_gene_mappings = {}
        self.species_to_rev_gene_mappings = {}
        for species,path in species_dict.items():
            
            if source.lower() == "plantcyc" or source.lower() == "pmn":
                df = self._get_plantcyc_pathway_dataframe(species,path)
                fwd_mapping, rev_mapping = self._get_plantcyc_pathway_gene_mappings(species, df)
            
            elif source.lower() == "kegg":
                df = self._get_kegg_pathway_dataframe(species)
                fwd_mapping, rev_mapping = self._get_kegg_pathway_gene_mappings(species, df)
            
            else:
                print("name of source ({}) not recognized, attempting to use PlantCyc".format(source))
                df = self._get_plantcyc_pathway_dataframe(species,path)
                fwd_mapping, rev_mapping = self._get_plantcyc_pathway_gene_mappings(species, df)

            self.species_to_df_dict[species] = df            
            self.species_to_fwd_gene_mappings[species] = fwd_mapping
            self.species_to_rev_gene_mappings[species] = rev_mapping





    def get_pathway_dict(self, species, gene_dict):
        membership_dict = {}
        for gene_id, gene_obj in gene_dict.items():
            membership_dict[gene_id] = self.get_pathway_membership(species=species, gene_obj=gene_obj)
        return(membership_dict)


    def get_pathway_membership(self, species, gene_obj):
        pathway_ids = []
        pathway_ids.extend(itertools.chain.from_iterable([self.species_to_rev_gene_mappings[species][name] for name in gene_obj.names]))
        return(pathway_ids)



    def get_pathway_ids_from_gene_name(self, species, gene_name):
        return(self.species_to_rev_gene_mappings[species][name])



    def get_gene_names_from_pathway_id(self, species, pathway_id):
        return(self.species_to_fwd_gene_mappings[species][pathway_id])












    def _get_plantcyc_pathway_dataframe(self, species_code, pathways_filepath):


        usecols = ["Pathway-id", "Pathway-name", "Reaction-id", "EC", "Protein-id", "Protein-name", "Gene-id", "Gene-name"]
        usenames = ["pathway_id", "pathway_name", "reaction_id", "ec_number", "protein_id", "protein_name", "gene_id", "gene_name"]
        renamed = {k:v for k,v in zip(usecols,usenames)}
        df = pd.read_table(pathways_filepath, usecols=usecols)
        df.rename(columns=renamed, inplace=True)
        df.fillna("", inplace=True)
        

        # Have to manually look for conventions to avoid mistakes with including gene names.
        # The string "unknown" is used for missing values, don't add this as a gene name.
        df.replace(to_replace="unknown", value="", inplace=True)
        

        df["gene_names"] = np.vectorize(phenolog.nlp.concatenate_with_bar_delim)(df["protein_id"], df["protein_name"], df["gene_id"], df["gene_name"])
        df["species"] = species_code
        df = df[["species", "pathway_id", "pathway_name", "gene_names", "ec_number"]]
        return(df)







    def _get_plantcyc_pathway_gene_mappings(self, species_code, pathways_df):
        
        pathway_dict_fwd = defaultdict(list)
        pathway_dict_rev = defaultdict(list)
        delim = "|"
        for row in pathways_df.itertuples():
            gene_names = row.gene_names.strip().split(delim)
            for gene_name in gene_names:
                pathway_dict_fwd[row.pathway_id].append(gene_name)
                pathway_dict_rev[gene_name].append(row.pathway_id)
        return(pathway_dict_fwd, pathway_dict_rev)












    def _get_kegg_pathway_dataframe(self, kegg_species_abbreviation):
        """
        Create a dictionary mapping KEGG pathways to lists of genes. Code is adapted from the example of
        parsing pathway files obtained through the KEGG REST API, which can be found here:
        https://biopython-tutorial.readthedocs.io/en/latest/notebooks/18%20-%20KEGG.html
        The specifications for those files state that the first 12 characeters of each line are reserved
        for the string which species the section, like "GENE", and the remainder of the line is for 
        everything else.
        Args:
            kegg_species_pathway_abbreviation (str): Species abbreviation string, see table of options.
        Returns:
            pandas.DataFrame: The dataframe containing all relevant information about all applicable KEGG pathways.
        """

        col_names = ["species", "pathway_id", "pathway_name", "gene_names", "ncbi_id", "uniprot_id", "ko_number", "ec_number"]
        df = pd.DataFrame(columns=col_names)

        pathway_dict_fwd = {}
        pathway_dict_rev = defaultdict(list)
        pathways = REST.kegg_list("pathway", kegg_species_abbreviation)
        pathway_ids_dict = {}

        ctr = 0

        for pathway in pathways:

            ctr = ctr+1
            if ctr>5:
                break

            pathway_file = REST.kegg_get(dbentries=pathway).read()
            for line in pathway_file.rstrip().split("\n"):
                section = line[:12].strip()
                if not section == "":
                    current_section = section



                # Collect information about the gene described on this line.
                if current_section == "GENE":

                    # Parse this line of the pathway file.
                    row_string = line[12:]
                    row_tokens = line[12:].split()
                    ncbi_accession = row_tokens[0]
                    uniprot_accession = ""

                    # Handing the gene names and other accessions with regex.
                    names_portion_without_accessions = " ".join(row_tokens[1:])
                    pattern_for_ko = r"(\[[A-Za-z0-9_|\.|:]*?KO[A-Za-z0-9_|\.|:]*?\])"
                    pattern_for_ec = r"(\[[A-Za-z0-9_|\.|:]*?EC[A-Za-z0-9_|\.|:]*?\])"
                    result_for_ko = re.search(pattern_for_ko, row_string)
                    result_for_ec = re.search(pattern_for_ec, row_string)
                    if result_for_ko == None:
                        ko_accession = ""
                    else:
                        ko_accession = result_for_ko.group(1)
                        names_portion_without_accessions = names_portion_without_accessions.replace(ko_accession, "")
                        ko_accession = ko_accession[1:-1]
                    if result_for_ec == None:
                        ec_accession = ""
                    else:
                        ec_accession = result_for_ec.group(1)
                        names_portion_without_accessions = names_portion_without_accessions.replace(ec_accession, "")
                        ec_accession = ec_accession[1:-1]

                    # Parse the other different names or symbols mentioned.
                    names = names_portion_without_accessions.split(";")
                    names = [name.strip() for name in names]
                    names_delim = "|"
                    names_str = names_delim.join(names)


                    # Update the dataframe no matter what the species was.
                    row = {
                        "species":kegg_species_abbreviation,
                        "pathway_id":pathway,
                        "pathway_name":pathway,
                        "gene_names":names_str,
                        "ncbi_id":ncbi_accession,
                        "uniprot_id":uniprot_accession,
                        "ko_number":ko_accession,
                        "ec_number":ec_accession
                    }
                    df = df.append(row, ignore_index=True, sort=False)



                # Update the dictionary between pathway names and IDs.
                if current_section == "KO_PATHWAY":
                    pathway_id = line[12:].strip()
                    pathway_ids_dict[pathway] = pathway_id


        # Update the pathway ID fields using the dictionary.
        df.replace({"pathway_id":pathway_ids_dict}, inplace=True)
        return(df)







    def _get_kegg_pathway_gene_mappings(self, kegg_species_abbreviation, kegg_pathways_df):
        """ Obtain forward and reverse mappings between pathways and gene names.
        Args:
            kegg_species_abbreviation (str): The species code for which genes to look at.
            kegg_pathways_df (pandas.DataFrame): The dataframe containing all the pathway information.
        Returns:
            (dict,dict): A mapping from pathway IDs to lists of gene names,
                         and a mapping from gene names to lists of pathway IDs. 
        """
        pathway_dict_fwd = defaultdict(list)
        pathway_dict_rev = defaultdict(list)
        delim = "|"
        for row in kegg_pathways_df.itertuples():
            gene_names = row.gene_names.strip().split(delim)
            if not row.ncbi_id == "":
                gene_names.append(phenolog.utils.add_tag(row.ncbi_id, phenolog.utils.ncbi_tag))
            if not row.uniprot_id == "":
                gene_names.append(phenolog.utils.add_tag(row.uniprot_id, phenolog.utils.uniprot_tag))
            for gene_name in gene_names:
                pathway_dict_fwd[row.pathway_id].append(gene_name)
                pathway_dict_rev[gene_name].append(row.pathway_id)
        return(pathway_dict_fwd, pathway_dict_rev)




















    def describe(self):
        print("\nDescribing the Pathways object...")
        print("Number of pathways found for each species:")
        for species in self.species_list:
            print("{}: {}".format(species, len(self.species_to_fwd_gene_mappings[species].keys())))
        print("Number of unique genes found mapped to pathways by species:")
        for species in self.species_list:
            query_str = "species=='{}'".format(species)
            print("{}: {}".format(species, len(self.species_to_df_dict[species].query(query_str))))
        print("Number of genes names found mapped to pathways by species:")
        for species in self.species_list:
            print("{}: {}".format(species, len(self.species_to_rev_gene_mappings[species].keys())))





















