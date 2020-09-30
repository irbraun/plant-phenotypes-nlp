library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)





# Read in the file that contains the all the precision and recall values for each method.
df <- read.csv(file="/Users/irbraun/Desktop/histograms.csv")
head(df)


# Using just a few particular examples to build the figure.
df <- df %>% filter((num_bins==50)) %>% filter((curated=="true")) %>% filter((approach=="doc2vec__wikipedia_size_300"))
head(df)


df <- df %>% filter(!((objective=="pathways") & (species!="both")))
head(df)






# Converting these variables to factors so that we can control the order in the plots.
positive_string <- "Gene pairs considered positive"
negative_string <- "All other gene pairs"
df$distribution <- factor(df$distribution, levels=c("positive","negative"), labels=c(positive_string,negative_string))
df$objective <- factor(df$objective, levels=c("orthologs","predicted","known","pathways","subsets"), labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))



# Pick colors to represent the positive and negative distributions.
dist_colors <- c(NA,"#0000007D")
dist_names <-  c(positive_string,negative_string)
color_mapping <- setNames(dist_colors, dist_names)


dist_colors <- c("#000000",NA)
dist_names <-  c(positive_string,negative_string)
color_mapping_2 <- setNames(dist_colors, dist_names)


ggplot(df, aes(x=bin_center, y=density, fill=distribution, color=distribution)) +
  geom_bar(stat="identity", position="identity", width=0.02, size=0.12) +
  theme_bw() +
  scale_fill_manual(name="Distribution", values=color_mapping) +
  scale_color_manual(name="Distribution", values=color_mapping_2) +
  scale_y_continuous(expand=c(0.00, 0.00)) +
  scale_x_continuous(expand=c(0.01, 0.00)) +
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
path <- "/Users/irbraun/Desktop/b.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=18, height=6, units=c("cm"), dpi=500, limitsize=FALSE)

