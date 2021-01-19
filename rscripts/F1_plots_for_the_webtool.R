library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(stringr)





# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/quoats/plots/combined_plot_data.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/tool_query_bargraphs.png"

# Reading in the data.
df <- read.csv(file=input_path, header=T, sep=",")
head(df)

# Converting variables to factors to control the formatting and order for the plot.
df$bin <- factor(df$bin, levels=c(10,20,30,40,50,100), labels=c("10","20","30","40","50",">50"))
df$from <- factor(df$from, levels=c("ath","zma"), labels=c("Arabidopsis","Maize"))
df$to <- factor(df$to, levels=c("ath","zma"), labels=c("Arabidopsis","Maize"))

df$header <- paste(as.character(df$from), "to", as.character(df$to))
df$header <- factor(df$header, 
                    levels=c("Arabidopsis to Arabidopsis","Maize to Maize", "Arabidopsis to Maize","Maize to Arabidopsis"),
                    labels=c("Arabidopsis to Arabidopsis","Maize to Maize", "Arabidopsis to Maize","Maize to Arabidopsis"))



df$facet <- factor(df$facet, 
                    levels=c("Autophagy (Arabidopsis)",
                             "Anthocyanin (Arabidopsis)",
                             "Anthocyanin (Maize)",
                             "Anthocyanin (Arabidopsis to Maize)",
                             "Anthocyanin (Maize to Arabidopsis)"),
                   labels=c("Autophagy Core (Arabidopsis)",
                            "Anthocyanin (Arabidopsis)",
                            "Anthocyanin (Maize)",
                            "Anthocyanin (Arabidopsis to Maize)",
                            "Anthocyanin (Maize to Arabidopsis)"))



# Make a plot.
ggplot(data=df, aes(x=bin, y=mean)) +
  geom_bar(stat="identity") + 
  geom_bar(stat="identity", color="black") +
  geom_errorbar(aes(ymin=mean, ymax=mean+sd), width=.3) +
  #facet_grid(rows=vars(to), cols=vars(from), scales="free") +
  facet_wrap(vars(facet), nrow=1, scales="free") +
  theme_bw() +
  
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        axis.text.y = element_text(hjust=0),
        legend.direction = "vertical", 
        legend.position = "none",
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), 
        axis.line=element_blank(),
        panel.spacing.x = unit(0.2, "lines"),
        panel.spacing.y = unit(0.2, "lines")) +
  ylab("Number of Genes") +
  xlab("Rank Bins")


# Saving the plot to a file.
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=33, height=7, units=c("cm"), dpi=350, limitsize=FALSE)








