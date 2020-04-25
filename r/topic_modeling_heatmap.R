
library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)



df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/phenome_subsets/part_5_topic_modeling.csv")

df_long <- gather(df, subset, value, c(-order,-topic), factor_key=TRUE)

df_long$topic = as.factor(df_long$topic)


high_color <- "black"
low_color <- "white"
legend_label <- "Topic Fraction"


ggplot(data=df_long, aes(x=reorder(subset,order), y=reorder(topic,-order))) + geom_tile(aes(fill=value), colour="black") +
  #scale_fill_gradient(low=low_color, high=high_color, name=legend_label, limits=c(0,.5), breaks=seq(0,.5,.25)) +
  scale_fill_gradient(low=low_color, high=high_color, name=legend_label) +
  theme_bw() +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), axis.line = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  ylab("Topic") +
  xlab("Phenotype Subset") + 
  theme(legend.position="right") +
  theme(legend.text = element_text(lineheight=1 ,hjust=0.5, size=6), legend.title =element_text(lineheight=1, hjust=0.5, size=7))


# Save the image of the plot.
path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=15, height=12, units=c("cm"), dpi=500, limitsize=FALSE)









# Second part that shows the similarity between each subset in terms of which genes were assigned it by Lloyd and Meinke (2012). Does it fit?

df <- read.csv(file="/Users/irbraun/Desktop/part_1_lmsubsets_similarity_matrix.csv")


# Create a mapping between subset name and order value.
map <- hashmap(as.character(df$group), df$order)
to_order <- function(x){return(map[[x]])}


df_long <- gather(df, group_2, value, c(-order, -group), factor_key=TRUE)
df_long$order_2 <- to_order(as.character(df_long$group_2))
df_long$group = as.factor(df_long$group)


high_color <- "black"
low_color <- "white"
legend_label <- "Topic Fraction"


ggplot(data=df_long, aes(x=reorder(group,order), y=reorder(group_2,-order_2))) + geom_tile(aes(fill=value), colour="black") +
  #scale_fill_gradient(low=low_color, high=high_color, name=legend_label, limits=c(0,.5), breaks=seq(0,.5,.25)) +
  scale_fill_gradient(low=low_color, high=high_color, name=legend_label) +
  theme_bw() +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), axis.line = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  ylab("Phenotype Subset") +
  xlab("Phenotype Subset") + 
  theme(legend.position="right") +
  theme(legend.text = element_text(lineheight=1 ,hjust=0.5, size=6), legend.title =element_text(lineheight=1, hjust=0.5, size=7))

# Save the image of the plot.
path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=15, height=12, units=c("cm"), dpi=500, limitsize=FALSE)


