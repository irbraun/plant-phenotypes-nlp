library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)



# Read in the file that contains the all the precision and recall values for each method.
df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/composition_09_29_2020_h12m25s57_6271/words_shared_by_species_melted.csv")
head(df)



# The order of how the factor levels are specified matters for the plot; this puts 0 on the left and 5 on the right.
df$others = factor(df$others, levels=c(5,4,3,2,1,0), labels=c("5","4","3","2","1","0"))


df$species = factor(df$species, 
                    levels=c("ath",
                             "zma",
                             "osa",
                             "mtr",
                             "gmx",
                             "sly"), 
                    labels=c("Arabidopsis",
                             "Maize",
                             "Rice",
                             "Medicago",
                             "Soybean",
                             "Tomato"))



# Pick from those colors to match the same number of values present in group_names.
method_colors <- c(
  "#581845",
  "#900C3F",
  "#C70039",
  "#F9483B",
  "#FED517",
  "#DAF7A6")
method_names <-  c("0","1","2","3","4","5")
color_mapping <- setNames(method_colors, method_names)






ggplot(df, aes(y=quantity, x=species, fill=others)) + 
  geom_bar(position="fill", stat="identity") +
  scale_fill_manual(name="Additional", values=color_mapping) +
  scale_y_continuous(expand=c(0.01, 0.01)) +
  coord_flip() +
  theme_bw() +  
  ylab("Proportion") +
  xlab("Species") +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(),
      legend.direction = "vertical",
      legend.position = "right")







