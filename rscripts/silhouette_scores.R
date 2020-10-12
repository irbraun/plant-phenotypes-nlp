library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)




PATH <- "~/Desktop/part_5_silhouette_scores_by_n.csv"
df <- read.csv(file=PATH, header=T, sep=",")



df_long <-  gather(df, method, value, c(-n), factor_key=TRUE)

df_long$aaa <- "method_category_placeholder"

ggplot(df_long, aes(x=n, y=value, group=method)) +
  geom_line(aes(color=aaa), alpha=0.5) +
  theme_bw()+
  ylab("Silhouette Score") +
  xlab("Number of Clusters") 




