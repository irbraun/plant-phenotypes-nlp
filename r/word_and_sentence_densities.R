library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(ggrepel)





sent_limit <- 40
word_limit <- 500


# Reading in the csv file and converting to long format.
df <- read.csv(file="~/Desktop/b.csv", header=T, sep=",")
before <- table(df$species)
df <- df[df["num_sents"]<sent_limit,]
df <- df[df["num_words"]<word_limit,]
after = table(df$species)
removed <- before-after
removed 

percent_removed <- (removed/before)*100
percent_removed


# df_long <- gather(df, data, value, num_words, num_sents, factor_key=TRUE)
# 
# # Generate the plot.
# ggplot(df_long, aes(x=value)) +
#   geom_density(alpha=0.4) +
#   #scale_fill_grey(name="Training Set") +
#   theme_bw() +
#   facet_wrap(facets=c("data","species"), scale="free") +
#   theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
#         panel.grid.major = element_blank(), 
#         panel.grid.minor = element_blank(),
#         axis.text.y = element_blank(),
#         axis.ticks.y = element_blank(),
#         legend.direction = "vertical",
#         legend.position = "right")+ 
#   ylab("Density") +
#   xlab("Number of Words")


# Generate the plot.
ggplot(df, aes(x=num_words,)) +
  geom_density(alpha=0.9, color="black", fill="darkgray") +
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

ggsave("~/Desktop/words.png", plot=last_plot(), device="png", path=NULL, scale=1, width=6, height=6, units=c("cm"), dpi=500, limitsize=FALSE)



# Generate the plot.
ggplot(df, aes(x=num_sents)) +
  geom_density(alpha=1.0, fill="gray") +
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

ggsave("~/Desktop/sents.png", plot=last_plot(), device="png", path=NULL, scale=1, width=6, height=6, units=c("cm"), dpi=500, limitsize=FALSE)

