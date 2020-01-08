library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)




# Specifying paths to input files and for saving figures.
infile_handpicked = "/Users/irbraun/phenologs-with-oats/data/scratch/phrase_pair_handpicked_results.csv"
infile_generalized = "/Users/irbraun/phenologs-with-oats/data/scratch/phrase_pair_generalized_results.csv"
outfile_handpicked = "~/Desktop/figure1.png"
outfile_generalized = "~/Desktop/figure2.png"






# Reading in the csv file and converting to long format.
df <- read.csv(file=infile_handpicked, header=T, sep=",")
df_long <- gather(df, method, value, Wikipedia, PubMed, factor_key=TRUE)

# Mapping training sets to colors.
names = c("Wikipedia","PubMed")
colors = c("white","black")
mapping = setNames(colors,names) 

# Generate the plot.
ggplot(df_long, aes(x=Pair,y=value)) + 
  geom_point(aes(fill=method), colour="black", pch=21, size=3) +
  coord_flip() +
  theme_bw() +
  scale_fill_manual(name="Training Set", values=mapping) +
  scale_x_discrete() +
  scale_y_continuous(breaks=seq(0,1,0.25), limits=c(0,1)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        legend.direction = "vertical", 
        legend.position = "right")+ 
  ylab("Distance Percentile") +
  xlab("Phrase Pairs")
  
# Save the image of the plot.
ggsave(outfile_handpicked, plot=last_plot(), device="png", path=NULL, scale=1, width=15, height=7, units=c("cm"), dpi=500, limitsize=FALSE)









# Reading in the csv file and converting to long format.
df <- read.csv(file=infile_generalized, header=T, sep=",")
df <- subset(df, N_Gram_raw==1.00)

df_long <- gather(df, method, value, Wikipedia, PubMed, factor_key=TRUE)

# Mapping training sets to colors.
names = c("Wikipedia","Pubmed")
colors = c("white","black")
mapping = setNames(colors,names) 

# Generate the plot.
ggplot(df_long, aes(x=value, fill=method)) +
  geom_density(alpha=0.4) +
  scale_fill_grey(name="Training Set") +
  theme_bw() +
  facet_grid(rows=vars(Ontology,Relationship),cols=vars(method)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Density") +
  xlab("Distance Percentiles")

# Save the image of the plot.
ggsave(outfile_generalized, plot=last_plot(), device="png", path=NULL, scale=1, width=15, height=7, units=c("cm"), dpi=500, limitsize=FALSE)










