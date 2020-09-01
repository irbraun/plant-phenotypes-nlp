library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)
library(operators)
library(stringr)



# Read in the file that contains the all the precision and recall values for each method.
df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/quick_09_01_2020_h12m14s19_1412/main_metrics/precision_recall_curves.csv")
head(df)






# SOME FILTERING STEPS, WILL HAVE TO BE CHANGED IF THE NAMING OR FORMAT OF COLUMNS IN THE INPUT FILE ARE CHANGED.




# Don't keep the automatically added precision=1 datapoint (upper-left corner), because it adds a meaningless straight line to that point.
df <- df %>% filter(precision != 1)

# We want to be able to differentiate between methods that used sentence tokenization and ones that didn't, because these make two very different groups in terms of score distribution.
df$tokenization <- str_detect(df$hyperparameters, "tokenization")

# However, we also want to include all the curated approaches in the plots that look at tokenized computational approaches, so duplicate those rows with a true value for tokenization.
# This way we can still use tokenization as a plot facet and retain the points from the curated approaches.
new_lines <- df %>% filter(group=="curated")
new_lines$tokenization <- T
df <- rbind(df, new_lines)

# Don't retain the points for the curated approaches when looking at plots for the full dataset that includes all the genes.
df <- df %>% filter( !((group=="curated") & (curated=="false")) )

# We don't need to look at all of the types of questions here, not interested in the inter-species and intra-species subsets for predicting pathways.
df <- df %>% 
  filter((task=="pathways" & species=="both") | (task!="pathways")) %>%
  filter((task!="orthologs")) %>%
  filter((approach!="mean"))







# SECTION THAT REQUIRES MANUALLY CHANGING THE STRINGS BASED ON HOW WE WANT THEM TO APPEAR IN THE FIGURE.


# Reworking some of the string columns so that they have the contents that should be seen in the plots, and in the right order.
df$curated = factor(df$curated, levels=c("true","false"), labels=c("With Annotations","All Text Data")) 
df$tokenization = factor(df$tokenization, levels=c(T,F), labels=c("Sentence Tokenized","Not Sentence Tokenized"))
df$task = factor(df$task, levels=c("predicted","known","pathways", "subsets"), labels=c("Predicted","Known","Pathways","Phenotypes")) 
df$approach = factor(df$approach, 
                   levels=c("doc2vec","word2vec","n-grams","topic modeling","noble coder","go","po","eqs"), 
                   labels=c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder","GO","PO","EQs"))



# Make a mapping between each approach and color hexcodes.
# Trick for getting a pallette of n colors that matches the default ggplot2 color scheme.
# Alpha value shoul be an integer between 0 and 256.
gg_color_hue <- function(n, alpha) {
  hues = seq(15, 375, length = n + 1)
  codes = hcl(h = hues, l = 65, c = 100)[1:n]
  codes <- paste(codes, as.hexmode(alpha), sep="")
  return(codes)
}

colorful_colors <- gg_color_hue(n=5, alpha=90)
grayscale_colors <- c("#000000", "#000000", "#000000")

# Pick from those colors to match the same number of values present in group_names.
method_colors <- c(colorful_colors, grayscale_colors)
method_names <-  c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder","GO","PO","EQs")
color_mapping <- setNames(method_colors, method_names)



# Make a mapping between each approach and linetypes.
method_linetypes <- c("solid", "solid", "solid", "solid", "solid", "solid", "dashed", "dotted")
method_names <- c("Doc2Vec","Word2Vec","N-Grams","Topic Modeling","NOBLE Coder","GO","PO","EQs")
linetype_mapping <- setNames(method_linetypes, method_names)












# A subset of the dataframe with just one row for each task, which can be used to map tasks to baseline AUC values.
baselines <- df[!duplicated(df[,c("task","curated")]),]


# A unique string for each combination of general approach and specific hyperparameters used.
df$name <- paste(as.character(df$approach), as.character(df$hyperparameters))

# Make the plot.
ggplot(df, aes(x=recall, y=precision, group=name, color=approach, linetype=approach)) + geom_line() +
  facet_grid(rows=vars(curated, tokenization), cols=vars(task)) +
  geom_hline(data=baselines, aes(yintercept=basline_auc), linetype="dashed", color="gray", size=0.5) +
  theme_bw() +
  scale_color_manual(name="", values=color_mapping) +
  scale_linetype_manual(name="", values=linetype_mapping) +
  scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  scale_x_continuous(breaks=c(0,0.25,0.5,0.75,1), limits=c(0,1), expand=c(0.00, 0.00)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        legend.position = "bottom",
        panel.spacing = unit(1.5, "lines")) +
  xlab("Recall") +
  ylab("Precision")




# Saving the plot to a file.
path <- "/Users/irbraun/Desktop/pr.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=24, height=14, units=c("cm"), dpi=300, limitsize=FALSE)



