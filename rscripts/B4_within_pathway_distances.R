library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)




prepare_dataframe <- function(input_path, num_shown, num_gene_threshold, y_axis_name, top){
  # Read in the input file with distances between all groups, and subset based on those parameters.
  df <- read.csv(file=input_path)
  df <- df %>% filter(n>=num_gene_threshold)
  
  
  
  
  
  ####### df <- df[1:min(num_shown,nrow(df)),]
  
  
  
  
  
  
  # Create a new string that combines the group name and the number of genes mapped to it in this dataset.
  df$name_for_plot = paste(df$full_name, " (n=", df$n, ")", sep="") 
  
  
  
  
  # Gather the dataframe into the long format, one value per row.
  #####   df_long <- gather(df, approach, avg_percentile, -group_id, -full_name, -name_for_plot, -n)
  ######head(df_long)
  
  
  df_long <- df
  
  # Merge with the naming file to add in the correct name and group for including in the figure.
  names_df <- read.csv(file=names_path, header=T, sep="\t")
  df_long <- merge(x=df_long, y=names_df, by.x=c("approach"), by.y=c("name_in_notebook"), all.x=TRUE)
  head(df_long)
  
  
  
  # Use this line to check and make sure there are no missing values. 
  # Missing values will be introduced in the previous step if some method names present in the input file are not in the naming file.
  # We want to get rid of methods that were not mentioned in the naming file for now in order to just disregard those.
  df_long[rowSums(is.na(df_long)) > 0,]
  df_long <- df_long %>% drop_na()
  
  if(top==F){
    num_shown <- -num_shown
  }
  
  
  # The baseline and curation methods are not applicable for this figure.
  df_long <- df_long %>% filter(!class %in% c("Curation","Baseline"))
    groups_to_show <- df_long %>% 
    select(group_id,percentile) %>% 
    group_by(group_id) %>% 
    summarize(avg_percentile=mean(percentile)) %>% 
    top_n(n=num_shown,wt=desc(avg_percentile)) %>% 
    pull(group_id)
  
  df_long <- df_long %>% filter(group_id %in% groups_to_show)
  
  
  
  
  df_long_t <- data.frame(df_long %>% group_by(group_id,class) %>% mutate(avg=min(percentile)))
  head(df_long_t)
  
  
  
  
  
  
  # Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
  #df_long_t <- data.frame(df_long %>% group_by(group_id,full_name,n,name_for_plot,class) %>% summarize(avg=min(percentile)))
  #head(df_long_t)
  
  df_long_t$facet <- y_axis_name
  
  return(df_long_t)
  
}












input_path_subsets <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_11_22_2020_h09m43s07_3025_plants/stacked_all_subsets_within_distances_melted.csv"
input_path_kegg <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_11_22_2020_h09m43s07_3025_plants/stacked_all_kegg_only_within_distances_melted.csv"
input_path_plantcyc <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_11_22_2020_h09m43s07_3025_plants/stacked_all_pmn_only_within_distances_melted.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/intragroup_distances.png"
names_path <- "/Users/irbraun/phenologs-with-oats/names.tsv"



# Parameters for what to include in the figure.
input_path <- input_path_subsets
num_shown = 50
num_gene_threshold = 3
width = 30
height_per_group = 0.35
y_axis_name = "Phenotype Category"
df_top <- prepare_dataframe(input_path, num_shown, num_gene_threshold, y_axis_name, T)




# Parameters for what to include in the figure.
input_path <- input_path_plantcyc
num_shown = 20
num_gene_threshold = 3
width = 30
height_per_group = 0.35
y_axis_name = "Biochemical Pathway (Top 20)"
df_middle <- prepare_dataframe(input_path, num_shown, num_gene_threshold, y_axis_name, T)


# Parameters for what to include in the figure.
input_path <- input_path_plantcyc
num_shown = 20
num_gene_threshold = 3
width = 30
height_per_group = 0.35
y_axis_name = "Biochemical Pathway (Bottom 20)"
df_bottom <- prepare_dataframe(input_path, num_shown, num_gene_threshold, y_axis_name, F)







df_long_t <- rbind(df_top, df_middle, df_bottom)


# Control the order of the facet grid.
df_long_t$facet = factor(df_long_t$facet, 
levels=c("Phenotype Category","Biochemical Pathway (Top 20)", "Biochemical Pathway (Bottom 20)"), 
labels=c("Phenotype Category","Biochemical Pathway (Top 20)", "Biochemical Pathway (Bottom 20)"))






# Pick colors for each bar.
num_colors_needed <- length(unique(df_long_t$method))

method_names <- c("Baseline",
                  "N-Grams",
                  "Annotation",
                  "N-Grams/Annotation",
                  "Topic Modeling",
                  "ML (Embeddings)",
                  "ML (Word Replacement)",
                  "ML (Max Similarity)",
                  "Curation")


method_colors <- c("#333333",   # black
                   "#F09250",   # reddish
                   "#f5cd1f",   # yellow
                   "#a4db89",    # green
                   "#a4db89",   # green
                   "#f5ac0f",   # orange
                   "#fdf49c",   # yellow orange
                   "#dbf859",   # neon green
                   "#DDDDDD")   #gray


method_colors <- c("#333333", 
                   "#a4db89", 
                   "#fdf49c", 
                   "#dbf859",
                   "#f5cd1f", 
                   "#f5ac0f", 
                   "#F09250", 
                   "#F09250", 
                   "#DDDDDD")
color_mapping <- setNames(method_colors, method_names)










outline_names <- c("Yes","No")
outline_colors <- c("#000000", "#FFFFFF00")
outline_mapping <- setNames(outline_colors, outline_names)

df_long_t$significant <- df_long_t$benjamini_hochberg
df_long_t$significant = factor(df_long_t$significant, levels=c("True","False"), labels=outline_names)




# Make the plot.
#ggplot(df_long_t, aes(x=avg, y=reorder(name_for_plot,-avg), fill=class, colour=significant)) + 
ggplot(df_long_t, aes(x=avg, y=reorder(name_for_plot,-avg), fill=class)) + 
  facet_grid(rows = vars(facet), scale="free", space="free") +
  scale_fill_manual(name="Approach Used", values=color_mapping) +
  #scale_color_manual(name="Significance",values=outline_mapping) +
  geom_point(pch=21, size=3, alpha=0.8) +
  theme_bw() +
  xlab("Intragroup Distance Percentile")  +
  ylab("")



# Save the plot as an image.
height = height_per_group*length(unique(df_long_t$name_for_plot))
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width, height=height, units=c("cm"), dpi=350, limitsize=FALSE)






