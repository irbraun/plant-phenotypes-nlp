library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)




# Reading in the CSV file of results.
PATH <- "~/Desktop/full_table.csv"
df <- read.csv(file=PATH, header=T, sep=",")




# Subset the dataframe based on a particular topic or dataset if necesssary.
#df <- subset(df, Topic=="Biochemical Pathways" & Data=="Entire Dataset")
#df <- subset(df, Topic=="Biochemical Pathways")


# Collapse the dataframe to average across all hyperparameters for each method and find the standard deviation.
map <- hashmap(as.character(df$Method), as.character(df$Group))
mapping_function <- function(x){return(map[[x]])}
df <- data.frame(df %>% group_by(Method) %>% summarize(auc_avg=mean(auc), auc_sd=sd(auc), Order=mean(Order)))
df[is.na(df)] <- 0.000 # Replace NA with 0 so that sd bars of height 0 can be added to bars for methods with no hyperparamter variation.
df$Category <- mapping_function(df$Method)



# Organizing the different methods presented into general categories.
df$group <- factor(df$Category, levels = c("NLP","Ensemble","Curated"))
group_colors <- c(grey.colors(n=3,start=0.1,end=1.0,alpha=1))
group_names <-  c("NLP","Ensemble","Curated")
group_mapping <- setNames(group_colors, group_names)




# Change these to change what values are used for plotting, so that the ggplot() call doesn't need to be changed.
baseline = read.csv(file=PATH, header=T, sep=",")$baseline[1]
y_lim <- 0.3
step_size <- 0.1
df$metric_to_use <- df$auc_avg
df$error_to_use <- df$auc_sd




ggplot(data=df, aes(x=reorder(Method,Order),y=metric_to_use,fill=group))+geom_bar(stat="identity") + geom_bar(stat="identity", color="black") + geom_errorbar(aes(ymin=metric_to_use, ymax=metric_to_use+error_to_use), width=.3) +
  #facet_grid(cols=vars(Data), rows=vars(Topic), scales="free") +
  theme_bw() +
  scale_fill_manual(name="Approach Used",values=group_mapping) +
  geom_abline(slope=0, intercept=baseline,  col = "red", lty=2) +
  scale_x_discrete(breaks=df$Method,labels=df$Method) +
  scale_y_continuous(breaks=seq(0,y_lim,step_size), limits=c(NA,y_lim)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
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
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=12, height=6, units=c("cm"), dpi=300, limitsize=TRUE)






