library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(stringr)





# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/stacked_01_16_2021_h11m36s10_2698_plants/stacked_f1_max.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/bargraph_with_points.png"
names_path <- "/Users/irbraun/phenologs-with-oats/names.tsv"




df <- read.csv(file=input_path, header=T, sep=",")
head(df)


# Gather all the different taks columns into one. 
# Rather than specifying all the task columns, this just ignore the ones that are for something else (-col).
df_long <- gather(df, task, value, -method, -hyperparameters, -order, -group, -name_key, -name_value)
head(df_long)




# THESE CHANGES SHOULD BE DONE IN TEH NOTEBOOK, GET RID OF UNUSED COLUMN, JUST NEED THE ONE NAME ONE.
# NEW THING CHECK THIS AND MOVE TO OTHER FIGURE SCRIPTS AS WELL.
# This should be done in the notebook!!!!! just have it write the full name with hyperparams instead.
#df_long$old <- paste(df_long$method, df$hyperparameters, sep="__")
df_long$old <- df_long$name_key
df_long <- subset(df_long, select = -c(order, method, hyperparameters, group))
head(df_long)



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


df_long_t <- df_long




# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$task <- factor(df_long_t$task, 
                         #levels=c("inter_all_orthologs","intra_all_predicted","intra_all_known","both_all_pathways","intra_all_subsets"), 
                         #labels=c("Orthologs","Predicted","Known","Pathways","Phenotypes"))
                         #levels=c("both_all_pathways","intra_all_subsets","both_curated_pathways","intra_curated_subsets"), 
                         #labels=c("Pathways (All Genes)","Phenotypes (All Genes)","Pathways (Curated)","Phenotypes (Curated)"))
                        #levels=c("intra_all_subsets","intra_curated_subsets"), 
                        #labels=c("Phenotypes (All Genes)","Phenotypes (Curated)"))
                        levels=c("intra_curated_subsets"), 
                        labels=c("Phenotypes"))
df_long_t <- df_long_t %>% drop_na(task)



# Convert the task column to a factor so that we can control the order and formattin and get rid of ones we don't want to plot.
df_long_t$tokenized <- factor(df_long_t$tokenized, 
                         levels=c(FALSE,TRUE), 
                         labels=c("Concatenated","Sentence Tokenized"))



# Pick colors for each bar.
num_colors_needed <- length(unique(df_long_t$method))

method_names <- c("Baseline",
                  "TF-IDF",
                  "Annotation",
                  "N-Grams/Annotation",
                  "Topic Modeling",
                  "ML (Embeddings)",
                  "ML (Word Replacement)",
                  "ML (Max Similarity)",
                  "Curation")


method_colors <- c("#333333",   # black
                   "#F09250",   # reddish
                   "#f5cd1f",   # yellow
                   "#a4db89",    # green
                   "#a4db89",   # green
                   "#f5ac0f",   # orange
                   "#fdf49c",   # yellow orange
                   "#dbf859",   # neon green
                   "#DDDDDD")   #gray
 

method_colors <- c("#333333", 
                   "#a4db89", 
                   "#fdf49c", 
                   "#dbf859",
                   "#f5cd1f", 
                   "#f5ac0f", 
                   "#F09250", 
                   "#F09250", 
                   "#DDDDDD")

method_colors <- c("#333333", 
                   "#FFFFFF", 
                   "#FFFFFF", 
                   "#FFFFFF",
                   "#FFFFFF", 
                   "#FFFFFF", 
                   "#FFFFFF", 
                   "#FFFFFF", 
                   "#DDDDDD")

color_mapping <- setNames(method_colors, method_names)



#df_long_t <- df_long_t %>% filter(!(method=="Curation" & task %in% c("Pathways (All Genes)","Phenotypes (All Genes)")))
df_long_t





# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
other <- data.frame(df_long_t %>% group_by(task,method,tokenized) %>% summarize(avg_value=mean(value), max_value=max(value), min_value=min(value), order=mean(order)))
head(other)




# Make a plot faceted by task that shows the general performance for each of the 
ggplot(data=NULL) +
  coord_flip() +
  scale_fill_manual(name="Approach", values=color_mapping) +
  geom_bar(data=other, aes(x=reorder(method,-order), y=max_value), color="black", stat="identity") +
  geom_bar(data=other, aes(x=reorder(method,-order), y=max_value, fill=method), stat="identity", alpha=1.0) +
  geom_point(data=df_long_t, aes(x=reorder(method,-order), y=value), color="black", alpha=0.9, size=2, shape=0) + 
  #facet_grid(rows=vars(tokenized), cols=vars(task)) +
  facet_grid(rows=vars(task), cols=vars(tokenized)) +
  scale_y_continuous(breaks=seq(0,0.6,0.1), limits=c(0,0.6), expand=c(0.00, 0.00)) +
  theme_bw() +
  
  # the colors and fills go here, see other files.
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        axis.text.y = element_text(hjust=0),
        legend.direction = "vertical", 
        legend.position = "none",
        #panel.grid.minor = element_blank(),
        #panel.grid.major = element_blank(),
        axis.line=element_blank(),
        panel.spacing.x = unit(1.0, "lines"),
        panel.spacing.y = unit(0.1, "lines")) +
  ylab("Maximum F1 Value") +
  xlab("Similarity Approach")



# Saving the plot to a file.
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=24, height=7, units=c("cm"), dpi=350, limitsize=FALSE)








# Make a plot faceted by task that shows the general performance for each of the 
#ggplot(data=df_long_t, aes(x=reorder(method,-order), y=value)) +
#  coord_flip() +
#  #scale_fill_manual(name="Approach", values=color_mapping) +
#  geom_point(stat="identity", color="black", alpha=0.6, size=2) + 
#  
#  geom_bar(stat="identity", color="black", alpha=0.2) +
#  
#  
#  facet_grid(rows=vars(tokenized), cols=vars(task)) +
#  scale_y_continuous(breaks=seq(0,0.6,0.1), limits=c(0,0.6), expand=c(0.00, 0.00)) +
#  theme_bw() +
#  
#  # the colors and fills go here, see other files.
#  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
#        axis.text.y = element_text(hjust=0),
#        legend.direction = "vertical", 
#        legend.position = "none",
#        #panel.grid.minor = element_blank(),
#        #panel.grid.major = element_blank(),
#        axis.line=element_blank(),
#        panel.spacing.x = unit(1.0, "lines"),
#        panel.spacing.y = unit(0.1, "lines")) +
#  ylab("Maximum F1 Value") +
#  xlab("Similarity Approach")


























