library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)




df1 <- read.csv(file="/Users/irbraun/Desktop/outputs/vec_and_curated/part_5_precision_recall_curves.csv")
df2 <- read.csv(file="/Users/irbraun/Desktop/outputs/nc_and_tm/part_5_precision_recall_curves.csv")
df3 <- read.csv(file="/Users/irbraun/Desktop/outputs/tm_and_vocab/part_5_precision_recall_curves.csv")
df4 <- read.csv(file="/Users/irbraun/Desktop/outputs/ngrams/part_5_precision_recall_curves.csv")
df <- rbind(df1, df2, df3, df4)
head(df)


df <- df[df["precision"] != 1,]


df$t <- str_detect(df$hyperparameters, "tokenization")


df <- df[df["t"] == FALSE,]
#df <- df[(df$method %in% c("go","po","eqs_distance")) | (df["t"] == TRUE) , ]




# CHANGE THESE TO ONLY REMOVE THE ONTOLOGY ONES FOR THE TOP

# Removing unused parts of the input dataframe.
df <- df[df["species"]=="both",]
df <- df[df["task"]!="orthologs",]

#df <- df[df$method %!in% c("go","po","eqs_distance","mean"), ]
df <- df[df$method %!in% c("mean"), ]



df <- df[(df$method %!in% c("go","po","eqs_distance")) | (df["curated"]=="true") , ]


# CHANGE THESE TO MAKE THE NLP APPROACHES JUST 'ONTOLOGY CURATION' OR "NLP"



# Specifying linetype stuff.
line.levels <- list(
  GO = "go",
  PO = "po",
  EQs = c("eqs_distance","doc2vec","word2vec","n-grams","topic modeling","noble coder")
)
df$line_type <- df$method
levels(df$line_type) <- line.levels





# Reworking some of the string columns so that they have the contents that should be seen in the plots, and in the right order.
df$curated = factor(df$curated, levels=c("true","false"), labels=c("With Annotations","All Text Data")) 
df$task = factor(df$task, levels=c("predicted","known","pathways", "subsets"), labels=c("Predicted","Known","Pathways","Phenotypes")) 
df$method = factor(df$method, 
                   levels=c("doc2vec","word2vec","n-grams","topic modeling","noble coder","go","po","eqs_distance"), 
                   labels=c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder","GO","PO","EQs"))







subset <- df











# Trick for getting a pallette of n colors that matches the default ggplot2 color scheme.
# Alpha value shoul be an integer between 0 and 256.
gg_color_hue <- function(n, alpha) {
  hues = seq(15, 375, length = n + 1)
  codes = hcl(h = hues, l = 65, c = 100)[1:n]
  codes <- paste(codes, as.hexmode(alpha), sep="")
  return(codes)
}

colorful_colors <- gg_color_hue(5, 90)

# Add alpha values to the ends of each color code.
#colorful_colors <- paste(colorful_colors, "64", sep="")


#grayscale_colors <- c(grey.colors(n=3,start=0.6,end=0.0,alpha=1))
grayscale_colors <- c("#000000", "#000000", "#000000")



# Pick from those colors to match the same number of values present in group_names.
method_colors <- c(colorful_colors, grayscale_colors)
method_names <-  c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder","GO","PO","EQs")
color_mapping <- setNames(method_colors, method_names)


a <- c("dashed","dotted","solid")
b <- c("GO","PO","EQs")
line_type_mapping <- setNames(a, b)





# A subset of the dataframe with just one row for each task, which can be used to mask tasks to baseline values.
baselines <- subset[!duplicated(subset[,c("task","curated")]),]


subset$name <- paste(as.character(subset$method), as.character(subset$hyperparameters))

ggplot(subset, aes(x=recall, y=precision, group=name, color=method, linetype=line_type)) + geom_line() +
  facet_grid(rows=vars(curated), cols=vars(task)) +
  geom_hline(data=baselines, aes(yintercept=basline_auc), linetype="dashed", color="gray", size=0.5) +
  theme_bw() +
  scale_color_manual(name="", values=color_mapping) +
  scale_linetype_manual(name="", values=line_type_mapping) +
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
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=24, height=14, units=c("cm"), dpi=300, limitsize=FALSE)



