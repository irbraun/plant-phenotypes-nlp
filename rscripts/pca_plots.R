library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)




df <- read.csv(file="/Users/irbraun/Desktop/b.csv")
df$cluster = as.factor(df$cluster)


ggplot(df, aes(x=component_1, y=component_2, color=cluster)) +
  geom_point(alpha=0.6)
  