library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(stringr)





# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_12_2020_h08m36s56_4218_plant/stacked_f1_max.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/bargraphs.png"





df <- read.csv(file=input_path, header=T, sep=",")
head(df)


# Gather all the different taks columns into one. 
# Rather than specifying all the task columns, this just ignore the ones that are for something else (-col).
df_long <- gather(df, task, value, -method, -hyperparameters, -order, -group)
head(df_long)





# We might want to break these into two groups 


df_long$tokenized <- str_detect(df_long$hyperparameters, "tokenization")


#df <- df[df["t"] == FALSE,]
#df <- df[(df$method %in% c("go","po","eqs_distance")) | (df["t"] == TRUE) , ]






# CHANGINT THIS HERE, WHATEVER GOES IN METHDO GETS USED
df_long$method <- df_long$hyperparameters




# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
df_long_t <- data.frame(df_long %>% group_by(task,method,tokenized) %>% summarize(avg_value=mean(value), max_value=max(value), min_value=min(value), order=mean(order)))
head(df_long_t)






# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$task <- factor(df_long_t$task, 
                         levels=c("inter_all_orthologs","intra_all_predicted","intra_all_known","both_all_pathways","intra_all_subsets"), 
                         labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))
df_long_t <- df_long_t %>% drop_na(task)






# Make a plot faceted by task that shows the general performance for each of the 
ggplot(data=df_long_t, aes(x=reorder(method,order), y=avg_value, fill=method)) +
  coord_flip() +
  geom_bar(stat="identity") + geom_bar(stat="identity", color="black") + 
  geom_errorbar(aes(ymin=min_value, ymax=max_value), width=.3) +
  facet_grid(rows=vars(tokenized), cols=vars(task)) +
  scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  theme_bw() +
  
  # the colors and fills go here, see other files.
  
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        axis.text.x = element_text(angle=60, vjust=1.0, hjust=1),
        axis.title.x = element_blank(),
        legend.direction = "vertical", 
        #legend.position = "right", 
        legend.position = "none",
        #panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.line=element_blank(),
        panel.spacing = unit(1.5, "lines")) +
  ylab("Maximum F2 Value") +
  xlab("Class of Approach")


# Saving the plot to a file.
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=20, height=8, units=c("cm"), dpi=300, limitsize=FALSE)








