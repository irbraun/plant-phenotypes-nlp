library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)




# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_12_2020_h08m36s56_4218_plant/stacked_histograms.csv"
output_dir <- "/Users/irbraun/phenologs-with-oats/figs/pairwise_distributions/"



# The output in this case is a folder that lots of dynamically named image files will be added to.
dir.create(file.path(output_dir))




# What about generating for a bunch of different combinations and saving them for a supplemental file?
# Function is just copied and pasted from the example above, with the fixed output path parameterized instead.
make_and_save_figure <- function(df, output_path){
  
  # Converting these variables to factors so that we can control the order in the plots.
  positive_string <- "Gene pairs considered positive"
  negative_string <- "All other gene pairs"
  df$distribution <- factor(df$distribution, levels=c("positive","negative"), labels=c(positive_string,negative_string))
  df$objective <- factor(df$objective, levels=c("orthologs","predicted","known","pathways","subsets"), labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))
  
  
  
  # Pick colors to represent the positive and negative distributions.
  dist_names <-  c(positive_string,negative_string)
  dist_colors_fill <- c(NA,"#0000007D")
  dist_colors_line <- c("#000000",NA)
  color_mapping_fill <- setNames(dist_colors_fill, dist_names)
  color_mapping_line <- setNames(dist_colors_line, dist_names)
  
  # Another way that uses just fill colors instead.
  #dist_names <-  c(positive_string,negative_string)
  #dist_colors_fill <- c("#F8766DC8","#06BFC4C8")
  #dist_colors_line <- c(NA,NA)
  #color_mapping_fill <- setNames(dist_colors_fill, dist_names)
  #color_mapping_line <- setNames(dist_colors_line, dist_names)
  
  y_axis_limit <- max(df$density)+2
  
  ggplot(df, aes(x=bin_center, y=density, fill=distribution, color=distribution)) +
    geom_bar(stat="identity", position="identity", width=0.02, size=0.12) +
    theme_bw() +
    scale_fill_manual(name="Distribution", values=color_mapping_fill) +
    scale_color_manual(name="Distribution", values=color_mapping_line) +
    scale_y_continuous(expand=c(0.00, 0.00)) + 
    expand_limits(y=y_axis_limit) +
    scale_x_continuous(expand=c(0.00, 0.00)) +
    facet_wrap(facets=vars(objective), nrow=1, scales="free") +
    theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.direction = "horizontal",
          legend.position = "bottom")+ 
    ylab("Density") +
    xlab("Gene Pair Similarity")
  
  # Saving the plot to a file.
  ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=18, height=6, units=c("cm"), dpi=500, limitsize=FALSE)
}


# Read in the dataframe and convert from distances to similarities for the purposes of these plots.
full_df <- read.csv(file=input_path)
head(full_df)
str(full_df)
full_df$bin_center = 1-full_df$bin_center


# Create one output file for each possible combination.
approaches_to_use <- unique(full_df$approach)
curation_subsets_to_use <- unique(full_df$curated)

# Or for testing purposes, just create one.
#approaches_to_use <- "n_grams__tokenization_full_nouns_adjectives_1_grams"
#curation_subsets_to_use <- "True"


# Create one set of plots faceted by task for each approach and subset of genes in the dataset.
for (a in approaches_to_use){
  for (c in curation_subsets_to_use){
    # Using just a few particular examples to build the figure.
    df <- (full_df %>% filter((num_bins==50)) %>% filter((curated==c)) %>% filter((approach==a)))
    df <- df %>% filter(!((objective=="pathways") & (species!="both")))
    output_path = paste(output_dir,a,"_curated_",c,".png", sep="")
    make_and_save_figure(df, output_path)
  }
}














