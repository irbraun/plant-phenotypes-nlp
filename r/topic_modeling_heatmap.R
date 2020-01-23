
library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)



df <- read.csv(file="/Users/irbraun/Desktop/hm.csv")

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

