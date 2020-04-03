library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(ggrepel)



PATH <- "~/Desktop/a.csv"
df <- read.csv(file=PATH, header=T, sep=",")
df$token <- as.character(df$token)








threshold_zma <- 0.6
threshold_ath <- 0.6
df$plotword <- ""
df[df$zma_rate>=threshold_zma,]$plotword <- df[df$zma_rate>=threshold_zma,]$token
df[df$ath_rate>=threshold_ath,]$plotword <- df[df$ath_rate>=threshold_ath,]$token


#df$group <- "both"
#df[df$zma_rate==0,]$group <- "ath"
#df[df$ath_rate==0,]$group <- "zma"






# Organizing the different methods presented into general categories.

#df$group <- factor(df$Category, levels = c("NLP","Embedding","Curated","Other"))
#group_colors_bw <- c(grey.colors(n=3,start=0.1,end=1.0,alpha=1))
#group_colors <- c("#581845", "#900C3F", "#C70039", "#FF5733", "#FFC300", "#DAF7A6", "#771142")

# Pick from those colors to match the same number of values present in group_names.
#group_colors <- c("#000000", group_colors[4], group_colors[5])
#group_names <-  c("both","ath","zma")
#group_mapping <- setNames(group_colors, group_names)







ggplot(df, aes(x=ath_rate, y=zma_rate)) +
  geom_point(alpha=0.6) +
  geom_text_repel(aes(label=plotword),color="black", hjust=-0.1, vjust=0.3) +
  theme_bw() +
  #scale_color_manual(name="Only",values=group_mapping) +
  scale_x_continuous(breaks=seq(0,2.0,0.5), limits=c(0,2.0), expand = c(0.01, 0)) +
  scale_y_continuous(breaks=seq(0,3.0,0.5), limits=c(0,3), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        #panel.grid.major = element_blank(), 
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Maize (Per 100 Words)") +
  xlab("Arabidopsis (Per 100 Words)")



# Save the image of the plot.
path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=20, height=20, units=c("cm"), dpi=300, limitsize=TRUE)




df <- df[df["ath_freq"]>0]



# What if we want the frequencies by which range they fit into?
# Have to make the numeric column categorical by binning.
bin_breaks <- c(-Inf, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, Inf)
bin_names <- c("low", "1","2","3","4","5","6", "7", "8", "9", "10-100",">100")


df$ath_freq_bin <- cut(df$ath_freq, breaks=bin_breaks, labels=bin_names)


head(df)



#df_a = df[df$ath_freq >0,]


# Generate the plot.
ggplot(df, aes(x=ath_freq_bin)) +
  geom_histogram(alpha=0.4, stat="count") +
  #scale_fill_grey(name="Training Set") +
  theme_bw() +
  #facet_grid(rows=vars(Ontology,Relationship),cols=vars(method)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        #axis.text.y = element_blank(),
        #axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Number of Words") +
  xlab("Total Word Frequency")
















