library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(ggrepel)



# The input and output files that this script uses and creates.
input_path <- "../data/scratch/word_sent_distributions.csv"
word_plot_output_path <-"../data/scratch/words.png"
sent_plot_output_path <- "../data/scratch/sents.png"



# This script creates two different faceted plot images that can then be combined later because they share the same
# ordering of ordering of the row facets, which is species. The input file should have three columns named species,
# num_words, and num_sents. The look of the output distribution plots is set here.
dpi = 500
width_cm = 10
height_cm = 10




# These values were hand-picked in order to remove <1% of the genes from just two of the species (and none from 
# the other species) while still making the plots very easy to interpret because these outliers have been removed.
# To verify this, just double-check the output of the number of genes removed and percent of genes removed for 
# each species when running this script, especially if changing these values. For this note, they are 40 and 500.
sent_limit <- 40
word_limit <- 500


# Reading in the csv file and converting to long format.
df <- read.csv(file=input_path, header=T, sep=",")
df$species = factor(df$species, levels=c('ath','zma','osa','sly','mtr','gmx'))
before <- table(df$species)
df <- df[df["num_sents"]<sent_limit,]
df <- df[df["num_words"]<word_limit,]
after = table(df$species)
number_removed <- before-after
number_removed 
percent_removed <- (number_removed/before)*100
percent_removed




# Generate the plot of word number distribution and save to a file.
ggplot(df, aes(x=num_words)) +
  geom_density(alpha=0.9, color="black", fill="lightgray") +
  theme_bw() +
  facet_grid(rows=vars(species),cols=vars(),scale="free") +
  scale_x_continuous(breaks=seq(0,word_limit,100), limits=c(0,word_limit+30), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Density") +
  xlab("Number of Words")

ggsave(word_plot_output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width_cm, height=height_cm, units=c("cm"), dpi=dpi, limitsize=FALSE)



# Generate the plot of sentence number distribution and save to a file.
ggplot(df, aes(x=num_sents)) +
  geom_density(alpha=1.0, fill="lightgray") +
  theme_bw() +
  facet_grid(rows=vars(species),cols=vars(),scale="free") +
  scale_x_continuous(breaks=seq(0,sent_limit,10), limits=c(0,sent_limit+2), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Density") +
  xlab("Number of Sentences")

ggsave(sent_plot_output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width_cm, height=height_cm, units=c("cm"), dpi=dpi, limitsize=FALSE)

