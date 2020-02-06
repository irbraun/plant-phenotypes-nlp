
library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)



df <- read.csv(file="/Users/irbraun/Desktop/part_6_within_distances.csv")


# Include the value for n with the full names.
df$name_for_plot = paste(df$full_name, " (n=", df$n, ")", sep="") 

# Only look at the top 50 pathways, if more were found.
df <- df[1:min(50,nrow(df)),]


ggplot(df, aes(x=mean_percentile, y=reorder(name_for_plot,-mean_percentile))) + 
  geom_point(colour="black", fill="black", pch=21, size=3) +
  theme_bw() +
  ylab("Biochemical Pathway") +
  xlab("Mean Within-Pathway Distance Percentile") 


path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=20, height=12, units=c("cm"), dpi=500, limitsize=FALSE)

