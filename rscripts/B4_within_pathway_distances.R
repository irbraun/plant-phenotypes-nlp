library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)





#input_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_14_2020_h08m41s58_6879_rev/stacked_subsets_within_distances.csv"
#input_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_14_2020_h08m41s58_6879_rev/stacked_kegg_only_within_distances.csv"
input_path = "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_14_2020_h08m41s58_6879_rev/stacked_pmn_only_within_distances.csv"


#output_path = "/Users/irbraun/phenologs-with-oats/figs/subsets_within_distances.png"
#output_path = "/Users/irbraun/phenologs-with-oats/figs/kegg_within_distances.png"
output_path = "/Users/irbraun/phenologs-with-oats/figs/plantcyc_within_distances.png"

names_path <- "/Users/irbraun/phenologs-with-oats/names.tsv"



# Parameters for what to include in the figure.
num_shown = 50
num_gene_threshold = 3
width = 30
height_per_group = 0.40
y_axis_name = "Biochemical Pathway"
#y_axis_name = "Phenotype Category"



# Read in the input file with distances between all groups, and subset based on those parameters.
df <- read.csv(file=input_path)
df <- df %>% filter(n>=num_gene_threshold)
df <- df[1:min(num_shown,nrow(df)),]



# Create a new string that combines the group name and the number of genes mapped to it in this dataset.
df$name_for_plot = paste(df$full_name, " (n=", df$n, ")", sep="") 




# Gather the dataframe into the long format, one value per row.
df_long <- gather(df, approach, avg_percentile, -group_id, -full_name, -name_for_plot, -n)
head(df_long)


# Merge with the naming file to add in the correct name and group for including in the figure.
names_df <- read.csv(file=names_path, header=T, sep="\t")
df_long <- merge(x=df_long, y=names_df, by.x=c("approach"), by.y=c("name_in_notebook"), all.x=TRUE)
head(df_long)



# Use this line to check and make sure there are no missing values. 
# Missing values will be introduced in the previous step if some method names present in the input file are not in the naming file.
# We want to get rid of methods that were not mentioned in the naming file for now in order to just disregard those.
df_long[rowSums(is.na(df_long)) > 0,]
df_long <- df_long %>% drop_na()

# The baseline and curation methods are not applicable for this figure.
df_long <- df_long %>% filter(!class %in% c("Curation","Baseline"))

head(df_long)





# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
df_long_t <- data.frame(df_long %>% group_by(group_id,full_name,n,name_for_plot,class) %>% summarize(avg=min(avg_percentile)))
head(df_long_t)



# Specifying the colors to represent each approach.
method_names <- c("Baseline",
                  "N-Grams",
                  "Topic Modeling",
                  "ML",
                  "N-Grams/ML",
                  "Annotation",
                  "N-Grams/Annotation",
                  "Curation")

method_colors <- c("#333333", "#F09250", "#fdf49c", "#a4db89", "#f5ac0f", "#f5cd1f", "#dbf859", "#DDDDDD")
color_mapping <- setNames(method_colors, method_names)




# Make the plot.
ggplot(df_long_t, aes(x=avg, y=reorder(name_for_plot,-avg), fill=class)) + 
  scale_fill_manual(name="Approach Used", values=color_mapping) +
  geom_point(colour="black", pch=21, size=3, alpha=0.8) +
  theme_bw() +
  ylab(y_axis_name) +
  xlab("Intragroup Distance Percentile") 



# Save the plot as an image.
height = height_per_group*nrow(df)
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width, height=height, units=c("cm"), dpi=400, limitsize=FALSE)






