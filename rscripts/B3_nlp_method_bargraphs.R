library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(stringr)





# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_10_14_2020_h08m41s58_6879_rev/stacked_f1_max.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/bargraphs.png"
names_path <- "/Users/irbraun/phenologs-with-oats/names.tsv"




df <- read.csv(file=input_path, header=T, sep=",")
head(df)


# Gather all the different taks columns into one. 
# Rather than specifying all the task columns, this just ignore the ones that are for something else (-col).
df_long <- gather(df, task, value, -method, -hyperparameters, -order, -group)
head(df_long)




# THESE CHANGES SHOULD BE DONE IN TEH NOTEBOOK, GET RID OF UNUSED COLUMN, JUST NEED THE ONE NAME ONE.
# NEW THING CHECK THIS AND MOVE TO OTHER FIGURE SCRIPTS AS WELL.
# This should be done in the notebook!!!!! just have it write the full name with hyperparams instead.
df_long$old <- paste(df_long$method, df$hyperparameters, sep="__")
df_long <- subset(df_long, select = -c(order, method, hyperparameters, group) )



names_df <- read.csv(file=names_path, header=T, sep="\t")

df_long <- merge(x=df_long, y=names_df, by.x=c("old"), by.y=c("name_in_notebook"), all.x=TRUE)

df_long[rowSums(is.na(df_long)) > 0,]
head(df_long)




# Some weird special cases here. We want to facet on tokenization, but also show curated approaches in both.
# But also exclude curated approaches for the entire dataset not subset by curation.
df_long$tokenized <- str_detect(df_long$tokenized, "yes")
extra_lines <- df_long[df_long$class=="Curation",]
extra_lines$tokenized <- TRUE
df_long <- rbind(df_long,extra_lines)





# Whatever data goes in the methods column will be used to define the bars in the plot.
df_long$method <- df_long$class


# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
df_long_t <- data.frame(df_long %>% group_by(task,method,tokenized) %>% summarize(avg_value=mean(value), max_value=max(value), min_value=min(value), order=mean(order)))
head(df_long_t)









# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$task <- factor(df_long_t$task, 
                         #levels=c("inter_all_orthologs","intra_all_predicted","intra_all_known","both_all_pathways","intra_all_subsets"), 
                         #labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))
                         #levels=c("both_all_pathways","intra_all_subsets","both_curated_pathways","intra_curated_subsets"), 
                         #labels=c("Pathways (All Genes)","Phenotypes (All Genes)","Pathways (Curated)","Phenotypes (Curated)"))
                          levels=c("intra_all_subsets","intra_curated_subsets"), 
                          labels=c("Phenotypes (All Genes)","Phenotypes (Curated)"))
df_long_t <- df_long_t %>% drop_na(task)



# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$tokenized <- factor(df_long_t$tokenized, 
                         levels=c(FALSE,TRUE), 
                         labels=c("Concatenated","Sentence Tokenized"))



# Pick colors for each bar.
num_colors_needed <- length(unique(df_long_t$method))

method_names <- c("Baseline",
                  "N-Grams",
                  "Topic Modeling",
                  "ML",
                  "N-Grams/ML",
                  "Annotation",
                  "N-Grams/Annotation",
                  "Curation")

method_colors <- c("#333333", "#F09250", "#fdf49c", "#a4db89", "#f5ac0f", "#f5cd1f", "#dbf859", "#DDDDDD")
color_mapping <- setNames(method_colors, method_names)










df_long_t <- df_long_t %>% filter(!(method=="Curation" & task %in% c("Pathways (All Genes)","Phenotypes (All Genes)")))
df_long_t










# Make a plot faceted by task that shows the general performance for each of the 
ggplot(data=df_long_t, aes(x=reorder(method,-order), y=avg_value, fill=method)) +
  coord_flip() +
  scale_fill_manual(name="Shared", values=color_mapping) +
  geom_bar(stat="identity") + geom_bar(stat="identity", color="black") + 
  geom_errorbar(aes(ymin=min_value, ymax=max_value), width=.3) +
  facet_grid(rows=vars(tokenized), cols=vars(task)) +
  #scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  scale_y_continuous(breaks=seq(0,0.6,0.1), limits=c(0,0.6), expand=c(0.00, 0.00)) +
  theme_bw() +
  
  # the colors and fills go here, see other files.
  
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        axis.text.y = element_text(hjust=0),
        #axis.text.x = element_text(angle=60, vjust=1.0, hjust=1),
        #axis.title.x = element_blank(),
        legend.direction = "vertical", 
        #legend.position = "right", 
        legend.position = "none",
        #panel.grid.minor = element_line(color="lightgray"), 
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        axis.line=element_blank(),
        panel.spacing.x = unit(1.0, "lines"),
        panel.spacing.y = unit(0.1, "lines")) +
  ylab("Maximum F1 Value") +
  xlab("Similarity Approach")


# Saving the plot to a file.
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=15, height=8, units=c("cm"), dpi=300, limitsize=FALSE)








