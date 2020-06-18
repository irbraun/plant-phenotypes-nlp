library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)




library(operators)




df <- read.csv(file="/Users/irbraun/Desktop/part_5_precision_recall_curves.csv")


# CHANGE THESE TO ONLY REMOVE THE ONTOLOGY ONES FOR THE TOP

# Removing unused parts of the input dataframe.
df <- df[df["species"]=="both",]
df <- df[df["task"]!="orthologs",]

df <- df[df$method %!in% c("go","po","eqs_distance","mean"), ]





# CHANGE THESE TO MAKE THE NLP APPROACHES JUST 'ONTOLOGY CURATION' OR "NLP"


# Reworking some of the string columns so that they have the contents that should be seen in the plots, and in the right order.
df$curated = factor(df$curated, levels=c("true","false"), labels=c("With Annotations","All Text Data")) 
df$task = factor(df$task, levels=c("predicted","known","pathways", "subsets"), labels=c("Predicted","Known","Pathways","Phenotypes")) 
df$method = factor(df$method, levels=c("doc2vec","word2vec","n-grams","topic modeling","noble coder"), labels=c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder")) 



subset <- df



group_colors <- c("#581845", "#900C3F", "#C70039", "#FF5733", "#FFC300", "#DAF7A6", "#771142")

# Pick from those colors to match the same number of values present in group_names.
method_colors <- c(group_colors[1], group_colors[2], group_colors[3], group_colors[4], group_colors[5])
method_names <-  c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder")
color_mapping <- setNames(method_colors, method_names)





# A subset of the dataframe with just one row for each task, which can be used to mask tasks to baseline values.
baselines <- subset[!duplicated(subset[,c("task")]),]


subset$name <- paste(as.character(subset$method), as.character(subset$hyperparameters))

ggplot(subset, aes(x=recall, y=precision, group=name, color=method)) + geom_line() +
  facet_grid(rows=vars(curated), cols=vars(task)) +
  geom_hline(data=baselines, aes(yintercept=basline_auc), linetype="dashed", color="gray", size=0.5) +
  theme_bw() +
  scale_color_manual(name="NLP Approach", values=color_mapping) +
  scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  scale_x_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        #panel.grid.major = element_blank(), 
        #panel.grid.minor = element_blank(), 
        legend.position = "bottom",
        panel.spacing = unit(1.5, "lines")) +
  xlab("Recall") +
  ylab("Precision")



# Saving the plot to a file.
path <- "/Users/irbraun/Desktop/pr.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=30, height=8, units=c("cm"), dpi=300, limitsize=FALSE)



