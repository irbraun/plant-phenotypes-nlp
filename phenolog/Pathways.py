from Bio.KEGG import REST
from collections import defaultdict
import pandas as pd
import re
import itertools





class Pathways:




    def __init__(self, species_list):

        # Create one dataframe and pair of dictionaries for each species code.
        self.species_to_df_dict = {}
        self.species_to_fwd_gene_mappings_by_name = {}
        self.species_to_rev_gene_mappings_by_name = {}
        self.species_to_fwd_gene_mappings_by_uniprot_id = {}
        self.species_to_rev_gene_mappings_by_uniprot_id = {}
        self.species_to_fwd_gene_mappings_by_ncbi_id = {}
        self.species_to_rev_gene_mappings_by_ncbi_id = {}
        for species in species_list:
            
            df = self._get_kegg_pathway_dataframe(species)
            self.species_to_df_dict[species] = df

            fwd_mapping, rev_mapping = self._get_kegg_pathway_gene_mappings(species, df, accession_type="name")
            self.species_to_fwd_gene_mappings_by_name[species] = fwd_mapping
            self.species_to_rev_gene_mappings_by_name[species] = rev_mapping

            fwd_mapping, rev_mapping = self._get_kegg_pathway_gene_mappings(species, df, accession_type="ncbi")
            self.species_to_fwd_gene_mappings_by_ncbi_id[species] = fwd_mapping
            self.species_to_rev_gene_mappings_by_ncbi_id[species] = rev_mapping

            fwd_mapping, rev_mapping = self._get_kegg_pathway_gene_mappings(species, df, accession_type="uniprot")
            self.species_to_fwd_gene_mappings_by_uniprot_id[species] = fwd_mapping
            self.species_to_rev_gene_mappings_by_uniprot_id[species] = rev_mapping







    def _get_kegg_pathway_gene_mappings(self, kegg_species_abbreviation, kegg_pathways_df, accession_type="name", delim="|"):
        """Summary
        
        Args:
            kegg_species_abbreviation (TYPE): Description
            kegg_pathways_df (TYPE): Description
            accession_type (str, optional): Description
            delim (str, optional): Description
        
        Returns:
            TYPE: Description
        """
        pathway_dict_fwd = defaultdict(list)
        pathway_dict_rev = defaultdict(list)

        # The default case, use gene names to create the mappings.
        if accession_type == "name":
            for row in kegg_pathways_df.itertuples():
                gene_names = row.gene_names.strip().split(delim)
                for gene_name in gene_names:
                    pathway_dict_fwd[row.pathway_id].append(gene_name)
                    pathway_dict_rev[gene_name].append(row.pathway_id)

        # Use NCBI accession numbers to create the mappings instead.
        if accession_type == "ncbi":
            for row in kegg_pathways_df.itertuples():
                if not row.gene_ncbi == "":
                    pathway_dict_fwd[row.pathway_id].append(row.gene_ncbi.strip())
                    pathway_dict_rev[row.gene_ncbi.strip()].append(row.pathway_id)

        # Use UniProt accession numbers to create the mappings instead.
        if accession_type == "uniprot":
            for row in kegg_pathways_df.itertuples():
                if not row.gene_uniprot == "":
                    pathway_dict_fwd[row.pathway_id].append(row.gene_uniprot.strip())
                    pathway_dict_rev[row.gene_uniprot.strip()].append(row.pathway_id)

        return(pathway_dict_fwd, pathway_dict_rev)







    def get_kegg_pathway_dict(self, species, gene_dict):
        """Summary
        
        Args:
            species (TYPE): Description
            gene_dict (TYPE): Description
        
        Returns:
            TYPE: Description
        
        Deleted Parameters:
            gene_list (TYPE): Description
        """
        membership_dict = {}
        for gene_id, gene_obj in gene_dict.items():
            membership_dict[gene_id] = self._get_kegg_pathway_membership(species=species, gene_obj=gene_obj)
        return(membership_dict)




    def _get_kegg_pathway_membership(self, species, gene_obj):
        """Use all available information to find the which pathways this gene belongs to.
        
        Args:
            species (TYPE): Description
            gene (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        pathway_ids = []
        pathway_ids.extend(itertools.chain.from_iterable([self.species_to_rev_gene_mappings_by_name[species][name] for name in gene_obj.names]))
        pathway_ids.extend(self.species_to_rev_gene_mappings_by_uniprot_id[species][gene_obj.uniprot_id])
        pathway_ids.extend(self.species_to_rev_gene_mappings_by_ncbi_id[species][gene_obj.ncbi_id])
        return(pathway_ids)







    def get_kegg_pathway_representation():
        return(None)











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
            TYPE: description
        """

        col_names = ["species", "pathway_id", "pathway_name", "gene_names", "gene_ncbi", "gene_uniprot", "gene_ko_number", "gene_ec_number"]
        df = pd.DataFrame(columns=col_names)

        pathway_dict_fwd = {}
        pathway_dict_rev = defaultdict(list)
        pathways = REST.kegg_list("pathway", kegg_species_abbreviation)
        pathway_ids_dict = {}

        ctr = 0

        for pathway in pathways:

            ctr = ctr+1
            if ctr>10000:
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
                        "gene_ncbi":ncbi_accession,
                        "gene_uniprot":uniprot_accession,
                        "gene_ko_number":ko_accession,
                        "gene_ec_number":ec_accession
                    }
                    df = df.append(row, ignore_index=True, sort=False)



                # Update the dictionary between pathway names and IDs.
                if current_section == "KO_PATHWAY":
                    pathway_id = line[12:].strip()
                    pathway_ids_dict[pathway] = pathway_id



        # Update the pathway ID fields using the dictionary.
        df.replace({"pathway_id":pathway_ids_dict}, inplace=True)
        return(df)



















