from Bio.KEGG import REST
from collections import defaultdict
import pandas as pd
import re







def get_kegg_pathway_dictionary_OLD(kegg_species_abbreviation):

    pathway_dict_fwd = {}
    pathway_dict_rev = defaultdict(list)
    pathways = REST.kegg_list("pathway", kegg_species_abbreviation)
    for pathway in pathways:
        pathway_file = REST.kegg_get(dbentries=pathway).read()
        gene_names = set()
        for line in pathway_file.rstrip().split("\n"):
            section = line[:12].strip()
            if not section == "":
                current_section = section

        # Update the forward and reverse dictionaries.
        pathway_dict_fwd[pathway] = gene_names
        for gene_name in gene_names:
            pathway_dict_rev[gene_name].append(pathway)

    return(pathway_dict_fwd, pathway_dict_rev)






def get_kegg_pathway_representation(pathways_df, gene_list):
    """Summary
    
    Args:
        pathways_df (TYPE): Description
        gene_list (TYPE): Description
    """
















def get_kegg_pathway_dataframe(kegg_species_abbreviation, path=None):
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

    col_names = ["species", "pathway_id", "pathway_name", "gene_name", "gene_ncbi", "gene_uniprot", "gene_ko_number", "gene_ec_number"]
    df = pd.DataFrame(columns=col_names)

    pathway_dict_fwd = {}
    pathway_dict_rev = defaultdict(list)
    pathways = REST.kegg_list("pathway", kegg_species_abbreviation)
    pathway_ids_dict = {}

    ctr = 0

    for pathway in pathways:

        ctr = ctr+1
        if ctr>10:
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
                    "gene_name":names_str,
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

    # Write the dataframe to a csv file if a path was provided.
    if not path == None:
        df.to_csv(path, index=False)

    return(df)



















