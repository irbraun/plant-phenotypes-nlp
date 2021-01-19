library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)




# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_01_16_2021_h11m36s10_2698_plants/stacked_precision_recall_curves.csv"
output_dir <- "/Users/irbraun/phenologs-with-oats/figs/precision_recall_curves/"


# The output in this case is a folder that lots of dynamically named image files will be added to.
dir.create(file.path(output_dir))



# What about generating for a bunch of different combinations and saving them for a supplemental file?
# Function is just copied and pasted from the example above, with the fixed output path parameterized instead.
make_and_save_figure <- function(df, baselines, output_path){
  
  
  # Don't keep the automatically added precision=1 datapoint (upper-left corner), because it adds a meaningless straight line to that point.
  df <- df %>% filter(precision != 1)
  
  
  # Reformatting to be a factor to specify spelling and re-ordering.
  df$task <- factor(df$task, levels=c("orthologs","known","pathways","subsets"), labels=c("Orthologs","Assocations","Pathways","Phenotypes"))
  baselines$task <- factor(baselines$task, levels=c("orthologs","known","pathways","subsets"), labels=c("Orthologs","Assocations","Pathways","Phenotypes"))
  df <- df %>% drop_na()
  baselines <- baselines %>% drop_na()
  
  
  
  # Make the plot.
  ggplot(df, aes(x=recall, y=precision)) + geom_line() +
    facet_wrap(facets=vars(task), nrow=1) +
    geom_hline(data=baselines, aes(yintercept=basline_auc), linetype="dashed", color="gray", size=0.5) +
    scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
    scale_x_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
    theme_bw() +
    xlab("Recall") +
    ylab("Precision") +
    theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5),
          panel.grid.major = element_blank(), 
          #panel.grid.minor = element_blank(),
          panel.spacing = unit(1.5, "lines"))
  
  # Saving the plot to a file.
  dim_for_one_plot <- 4
  height <- dim_for_one_plot+1
  width <- dim_for_one_plot*(length(unique(df$task)))
  
  ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width, height=height, units=c("cm"), dpi=500, limitsize=FALSE)
}





# Read in the full dataframe.
full_df <- read.csv(file=input_path)


# Create one output file for each possible combination.
approaches_to_use <- unique(full_df$name)
curation_subsets_to_use <- unique(full_df$curated)

# Or for testing purposes, just create one.
#approaches_to_use <- "n_grams__tokenization_full_nouns_adjectives_1_grams"
#curation_subsets_to_use <- "True"


# Create one set of plots faceted by task for each approach and subset of genes in the dataset.
for (a in approaches_to_use){
  for (c in curation_subsets_to_use){
    # Using just a few particular examples to build the figure. Create a baselines dataframe as well.
    df <- (full_df %>% filter((curated==c)) %>% filter((name==a)))
    df <- df %>% filter(!((task=="pathways") & (species!="both")))
    baselines <- df[!duplicated(df[,c("task","curated")]),]
    output_path = paste(output_dir,a,"_curated_",tolower(as.character(c)),".png", sep="")
    make_and_save_figure(df, baselines, output_path)
  }
}






