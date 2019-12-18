library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)




# Reading in the CSV file of results.
path <- "~/Downloads/Phenologs Paper Tables - Sheet4 (3).csv"
df <- read.csv(file=path, header=T, sep=",")


# Organizing the different methods presented into general categories.
df$group <- factor(df$Category, levels = c("NLP","Ensemble","Curation"))
group_colors <- c(grey.colors(n=3,start=0.15,end=0.95))
group_names <-  c("NLP","Ensemble","Curation")
group_mapping <- setNames(group_colors, group_names)



# Options for the plot.
y_lim <- 1.00
step_size <- 0.25
df$y_percent <- df$y_ratio*(100)


df <- subset(df, Topic=="Biochemical Pathways" & Data=="Entire Dataset")
#df <- subset(df, Topic=="Biochemical Pathways")

baseline = df$baseline[1]

ggplot(data=df, aes(x=reorder(Method,Order),y=auc,fill=group))+geom_bar(stat="identity") + geom_bar(stat="identity", color="black") +
  #facet_grid(cols=vars(Data), rows=vars(Topic), scales="free") +
  theme_bw() +
  scale_fill_manual(name="Approach Used",values=group_mapping) +
  geom_abline(slope=0, intercept=baseline,  col = "red", lty=2) +
  scale_x_discrete(breaks=df$Method,labels=df$Method) +
  scale_y_continuous(breaks=seq(0,y_lim,step_size), limits=c(NA,y_lim)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        #axis.text.x = element_text(angle=90, vjust=0.5, hjust=1),
        axis.text.x = element_text(angle=60, vjust=1.0, hjust=1),
        axis.title.x = element_blank(),
        legend.direction = "vertical", 
        legend.position = "right", 
        panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.major=element_blank(), 
        axis.line=element_blank()) +
  ylab("AUC")


# Save the image of the plot.
path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=17, height=5, units=c("cm"), dpi=300, limitsize=TRUE)














ggplot(data=df, aes(x=reorder(Method,order),y=auc,fill=group))+geom_bar(stat="identity") + geom_bar(stat="identity", color="black") +
  theme_bw() +
  scale_fill_manual(name="Approach Used",values=group_mapping) +
  scale_x_discrete(breaks=df$Method,labels=df$Method) +
  scale_y_continuous(breaks=seq(0,y_lim,step_size), limits=c(NA,y_lim)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        #axis.text.x = element_text(angle=90, vjust=0.5, hjust=1),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        legend.direction = "vertical", 
        legend.position = "right", 
        panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.major=element_blank(), 
        axis.line=element_blank()) +
  ylab("AUC")









# Reading in the CSV file of results.
path <- "~/Desktop/lloyd_table.csv"
df <- read.csv(file=path, header=T, sep=",")


# Organizing the different methods presented into general categories.
df$group <- factor(df$group, levels = c("Bag-of-words","N-grams","Embedding","Ontology","Ensemble","Curated"))
group_colors <- c(viridis(4,begin=0.2,end=0.95), grey.colors(n=2,start=0.15,end=0.95))
group_names <-  c("Bag-of-words","N-grams","Embedding","Ontology","Ensemble","Curated")
group_mapping <- setNames(group_colors, group_names)


# Options for the plot.
y_lim <- 1.00
step_size <- 0.25
df$y_percent <- df$y_ratio*(100)


ggplot(data=df, aes(x=reorder(method,order),y=auc,fill=group))+geom_bar(stat="identity") + geom_bar(stat="identity", color="black") +
  theme_bw() +
  scale_fill_manual(name="Approach Used",values=group_mapping) +
  scale_x_discrete(breaks=df$method,labels=df$method) +
  scale_y_continuous(breaks=seq(0,y_lim,step_size), limits=c(NA,y_lim)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        #axis.text.x = element_text(angle=90, vjust=0.5, hjust=1),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        legend.direction = "vertical", 
        legend.position = "right", 
        panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.major=element_blank(), 
        axis.line=element_blank()) +
  ylab("AUC")
  

# Save the image of the plot.
path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=17, height=5, units=c("cm"), dpi=300, limitsize=TRUE)

















