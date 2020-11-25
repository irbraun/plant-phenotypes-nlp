library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(ggrepel)



# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/composition_11_22_2020_h12m47s02_5791/word_sent_distributions.csv"
word_plot_output_path <- "/Users/irbraun/phenologs-with-oats/figs/word_distributions.png"
sent_plot_output_path <- "/Users/irbraun/phenologs-with-oats/figs/sent_distributions.png"





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
sent_limit <- 50
word_limit <- 500


# Reading in the csv file and converting to long format.
df <- read.csv(file=input_path, header=T, sep=",")
df$species <- factor(df$species, levels=c('ath','zma','osa','sly','mtr','gmx'))
before <- table(df$species)
df <- df[df["num_sents"]<=sent_limit,]
df <- df[df["num_words"]<=word_limit,]
after <- table(df$species)
number_removed <- before-after
number_removed 
percent_removed <- (number_removed/before)*100
percent_removed




# Generate the plot of word number distribution and save to a file.
num_bins = 30
right_padding = 0.0125 * word_limit
ggplot(df, aes(x=num_words)) +
  geom_histogram(alpha=0.9, color="black", fill="lightgray", breaks=seq(0,word_limit,word_limit/num_bins)) +
  theme_bw() +
  facet_grid(rows=vars(species),cols=vars(),scale="free") +
  scale_x_continuous(breaks=seq(0,word_limit,100), limits=c(0,word_limit+right_padding), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Frequency") +
  xlab("Number of Words")

ggsave(word_plot_output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width_cm, height=height_cm, units=c("cm"), dpi=dpi, limitsize=FALSE)




# Generate the plot of sentence number distribution and save to a file.
num_bins = 30
right_padding = 0.0125 * sent_limit
ggplot(df, aes(x=num_sents)) +
  geom_histogram(alpha=0.9, color="black", fill="lightgray", breaks=seq(0,sent_limit,sent_limit/num_bins)) +
  theme_bw() +
  facet_grid(rows=vars(species),cols=vars(),scale="free") +
  scale_x_continuous(breaks=seq(0,sent_limit,10), limits=c(0,sent_limit+right_padding), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Frequency") +
  xlab("Number of Sentences")

ggsave(sent_plot_output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=width_cm, height=height_cm, units=c("cm"), dpi=dpi, limitsize=FALSE)

