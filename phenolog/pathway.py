from Bio.KEGG import REST
from collections import defaultdict







def get_kegg_pathway_dictionary(kegg_species_pathway_abbreviation):
    """
    Create a dictionary mapping KEGG pathways to lists of genes. Code is adapted from the example of
    parsing pathway files obtained through the KEGG REST API, which can be found here:
    https://biopython-tutorial.readthedocs.io/en/latest/notebooks/18%20-%20KEGG.html
    The specifications for those files state that the first 12 characeters of each line are reserved
    for the string which species the section, like "GENE", and the remainder of the line is for 
    everything else. The formatting for the lines in the GENE section is not consistent between
    species, so some udpates have to ensure that this works with each new species added to a dataset.
    The gene_names set should include strings in whatever format is useful for returning.
    Args:
        kegg_species_pathway_abbreviation (str): Species abbreviation string, see table of options.
    Returns:
        (dict,dict): Mapping between strings which are pathway names and sets of strings of gene names.
                     Mapping between strings which are gene names and list of strings which are pathways.
    """
    pathway_dict_fwd = {}
    pathway_dict_rev = defaultdict(list)
    pathways = REST.kegg_list("pathway", kegg_species_pathway_abbreviation)
    for pathway in pathways:
        pathway_file = REST.kegg_get(dbentries=pathway).read()
        gene_names = set()
        for line in pathway_file.rstrip().split("\n"):
            section = line[:12].strip()
            if not section == "":
                current_section = section
            if current_section == "GENE":



                # Human
                if kegg_species_pathway_abbreviation == "hsa":
                    try:
                        gene_identifier, gene_description = line[12:].split("; ")
                        gene_symbol = gene_identifier
                        gene_names.add(gene_symbol)
                    except ValueError:
                        print("Irregular line in KEGG pathway file: ",line)


                # Arabidopsis thaliana
                if kegg_species_pathway_abbreviation == "ath":
                    try:
                        row_values = line[12:].split()
                        locus_name = row_values[0]
                        if ";" in row_values[1]:
                            gene_symbol = row_values[1]
                            gene_description = " ".join(line[2:]).strip()
                        else:
                            gene_symbol = ""
                            gene_description = " ".join(line[1:]).strip()
                        gene_names.add(locus_name)
                    except ValueError:
                        print("Irregular line in KEGG pathway file: ",line)



                # Next species...



        # Update the forward and reverse dictionaries.
        pathway_dict_fwd[pathway] = gene_names
        for gene_name in gene_names:
            pathway_dict_rev[gene_name].append(pathway)

    return(pathway_dict_fwd, pathway_dict_rev)













def get_kegg_pathway_dataframe(kegg_species_pathway_abbreviation):
    return(None)



















