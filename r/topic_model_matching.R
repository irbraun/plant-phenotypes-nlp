

library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)



df <- read.csv(file="/Users/irbraun/phenologs-with-oats/data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv")
subset_to_class_hashmap <- hashmap(as.character(df$Subset.Symbol), as.character(df$Class.Symbol))
subset_to_class <- function(x){return(subset_to_class_hashmap[[x]])}

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

class_to_color_hashmap <- hashmap(as.character(unique(df$Class.Symbol)), gg_color_hue(11))
class_to_color <- function(x){return(class_to_color_hashmap[[x]])}





df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/phenome_subsets/part_5_topic_modeling.csv")
get_col_index <- function(x){return(which(colnames(df)==(x)))}
df_long <- gather(df, subset, value, c(-order, -topic), factor_key=TRUE)
df_long$topic_order <- df_long$order
df_long$subset_order <- lapply(df_long$subset, get_col_index)
df_long$subset_order <- as.numeric(df_long$subset_order)
df_long$subset_order <- df_long$subset_order-3







df <- df_long
df$label_left <- df$subset
df$label_right <- as.factor(df$topic)
df$x <- as.integer(0)
df$xend <- as.integer(10)
df$y <- as.integer(df$subset_order)
df$yend <- df$topic_order
df$alpha <- df$value
df$size <- 0.4
str(df)

df$color <- class_to_color(subset_to_class(df$subset))






df <- df[df$alpha > .02, ]


range01 <- function(x){(x-min(x))/(max(x)-min(x))}
df$alpha <- range01(df$alpha)
df$alpha <- df$alpha/10




#df <- read.csv(file="/Users/irbraun/Desktop/c.csv")
#str(df)



#inherit.aes = FALSE

pl <- ggplot(df, aes(x=c(x,xend), y=c(y,yend)))


numbers <- unique(df$y)
left_hashmap<- hashmap(df$y, as.character(df$label_left))
right_hashmap <- hashmap(df$yend, as.character(df$label_right))
num_to_left_label <- function(x){return(left_hashmap[[x]])}
num_to_right_label <- function(x){return(right_hashmap[[x]])}




for (i in 1:nrow(df)){
#for (i in 1:400){
  pl <- pl + geom_segment(aes_string(x=df[i,"x"], y=df[i,"y"], xend=df[i,"xend"], yend=df[i,"yend"]), "size"=df[i,"alpha"]*10, "color"=df[i, "color"])
}

pl + 
  scale_y_continuous(name="Phenotypic Subset", breaks=numbers, labels=num_to_left_label, expand=c(0.01, 0.01),
                     sec.axis = sec_axis(~. , name = "Topic", breaks=numbers, labels=num_to_right_label)) +
  theme_bw() +
  
  scale_x_continuous(breaks=seq(0,10), limits=c(0,10), expand=c(0.01, 0.01)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
  
  

path <- "/Users/irbraun/Desktop/matching.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=8, height=20, units=c("cm"), dpi=500, limitsize=FALSE)








# ggplot() + 
#   geom_segment(aes(x = 0, y = 50, xend = 10, yend = 45, color="#000000", alpha=0.7, size=1)) +
#   geom_segment(aes(x = 0, y = 30, xend = 10, yend = 32, color="#0101F1", alpha=0.7, size=2)) +
#   geom_segment(aes(x = 0, y = 50, xend = 10, yend = 45, color="#000000", alpha=0.7, size=1)) +
#   geom_segment(aes(x = 0, y = 30, xend = 10, yend = 32, color="#0101F1", alpha=0.7, size=1))
