library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)




# Read in the data for both the PATO and PO specific files and stack them.
df1 <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/analysis_ontologies_run_for_pato/pato_distance_percentiles_all.csv", header=T, sep=",")
df2 <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/analysis_ontologies_run_for_po/po_distance_percentiles_all.csv", header=T, sep=",")
df <- rbind(df1,df2)




# Create a mapping between column names and how method names should look in the plots.
method_keys = c("Doc2Vec.Wikipedia", "Doc2Vec.PubMed", "BERT", "BioBERT", "N.Grams")
method_values = c("Doc2Vec Wikipedia", "Doc2Vec PubMed", "BERT", "BioBERT", "n-grams")
map1 <- hashmap(method_keys, method_values)
to_method_name <- function(x){return(map1[[x]])}

# Create a mapping between ontology edge names and how they should look in the plots.
edge_keys = c("parent_child", "sibling")
edge_values = c("Parent/Child", "Siblings")
map2 <- hashmap(edge_keys, edge_values)
to_edge_name <- function(x){return(map2[[x]])}



# Create a mapping between the method names used and broader categories to color the plots.
category_keys = c("Doc2Vec Wikipedia", "Doc2Vec PubMed", "BERT", "BioBERT", "n-grams")
category_values = c("Embedding", "Embedding", "Embedding", "Embedding", "N-Grams")
map3 <- hashmap(category_keys, category_values)
to_category <- function(x){return(map3[[x]])}

# What colors should be used for these categories?
category_colors <- c(grey.colors(n=2,start=0.0,end=1.0,alpha=1))
category_names <-  c("Embedding", "N-Grams")
category_color_mapping <- setNames(category_colors, category_names)









# Gather data into the long format and do renaming on the variables that will be facets.
df_long <- gather(df, method, value, method_keys, factor_key=TRUE)
df_long$method <- as.character(df_long$method)
df_long$method <- to_method_name(df_long$method)
df_long$category <- to_category(df_long$method)
df_long$relationship <- to_edge_name(df_long$relationship)


# Convert these variables into factors in order to specify the order they appear.
df_long$method <- factor(df_long$method, levels=method_values)
df_long$category <- factor(df_long$category, levels=category_names)





ggplot(df_long, aes(x=value, fill=category)) +
  geom_density(alpha=0.4) +
  scale_fill_manual(name="Categories",values=category_color_mapping) +
  theme_bw() +
  facet_grid(rows=vars(ontology,relationship),cols=vars(method)) +
  scale_x_continuous(breaks=seq(0,0.9999,0.5), limits=c(NA,NA)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Density") +
  xlab("Distance Percentiles")






