library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)




# Reading in the CSV file of results.
PATH <- "/Users/irbraun/Desktop/outputs/stacked_09_10_2020_h10m47s58_3118/stacked_f1_max.csv"
df <- read.csv(file=PATH, header=T, sep=",")
head(df)


# Gather all the different taks columns into one.
# NOTE: Check columns that are gathered here, shouldn't have to manually specify them like this.
df_long <- gather(df, task, value, intra_all_known:both_all_pathways)
head(df_long)


df_long$method <- df_long$group




# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
df_long_t <- data.frame(df_long %>% group_by(task,method) %>% summarize(avg_value=mean(value), max_value=max(value), min_value=min(value), order=mean(order)))
head(df_long_t)











# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$task <- factor(df_long_t$task, 
                         levels=c("inter_all_orthologs","intra_all_predicted","intra_all_known","both_all_pathways","intra_all_subsets"), 
                         labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))
df_long_t <- df_long_t %>% drop_na(task)






# Make a plot faceted by task that shows the general performance for each of the 
ggplot(data=df_long_t, aes(x=reorder(method,order), y=avg_value, fill=method)) + 
  geom_bar(stat="identity") + geom_bar(stat="identity", color="black") + 
  geom_errorbar(aes(ymin=min_value, ymax=max_value), width=.3) +
  facet_wrap(facets=vars(task), nrow=1) +
  theme_bw() +
  
  # the colors and fills go here, see other files.
  
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        axis.text.x = element_text(angle=60, vjust=1.0, hjust=1),
        #axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        legend.direction = "vertical", 
        #legend.position = "right", 
        legend.position = "none",
        panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.major=element_blank(), 
        axis.line=element_blank()) +
  ylab("Maximum F1 Value") +
  xlab("Class of Approach")








