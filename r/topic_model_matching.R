library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)


# Reading in the functional hierarchy file and creating some relevant hashmaps.
df <- read.csv(file="/Users/irbraun/phenologs-with-oats/data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv")
subset_to_class_hashmap <- hashmap(as.character(df$Subset.Symbol), as.character(df$Class.Symbol))
subset_to_class <- function(x){return(subset_to_class_hashmap[[x]])}
subset_to_description_hashmap <- hashmap(as.character(df$Subset.Symbol), as.character((df$Subset.Name.and.Description)))
subset_to_description <- function(x){return(subset_to_description_hashmap[[x]])}




# Trick for getting a pallette of n colors that matches the default ggplot2 color scheme.
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
class_to_color_hashmap <- hashmap(as.character(unique(df$Class.Symbol)), gg_color_hue(11))
class_to_color <- function(x){return(class_to_color_hashmap[[x]])}




# Reading in the matrix of subset to topic edge weights.
df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/05_15_2020_h14m49s25/part_6_topic_modeling.csv")
get_col_index <- function(x){return(which(colnames(df)==(x)))}
df_long <- gather(df, subset, value, c(-order, -topic, -tokens), factor_key=TRUE)
df_long$topic_order <- df_long$order
df_long$subset_order <- lapply(df_long$subset, get_col_index)
df_long$subset_order <- as.numeric(df_long$subset_order)
df_long$subset_order <- df_long$subset_order-3




# Setting the ranges for alpha and size for the lines that will appear in the plot based on edge weights.
weight_threshold <- 0.003
min_alpha <- 0.1
max_alpha <- 0.9
min_size <- 0.0001
max_size <- 1.0000


linear_conversion <- function(x, new_min, new_max, old_min, old_max){return((((x - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min)}


df <- df_long









# The right and left labels are string manipulation functions of some of other columns.
df$label_left <- paste(df$subset, " (", tolower(subset_to_description(df$subset)), ")", sep="")
df$label_right <- as.factor(paste(df$topic, " (", gsub('\\|', ', ', df$tokens), ")", sep=""))
df$x <- as.integer(0)
df$xend <- as.integer(10)
df$y <- as.integer(df$subset_order)
df$yend <- as.integer(df$topic_order+1)

# Re-scaling the alpha and size parameters for each line as linear functions of the weight ranges.
min_weight <- min(df$value)
max_weight <- max(df$value)
df$alpha <- linear_conversion(df$value, min_alpha, max_alpha, min_weight, max_weight)
df$size <- linear_conversion(df$value, min_size, max_size, min_weight, max_weight)
str(df)

# Setting the color of each line to depend on the source class (middle level in the functional hierarchy).
df$color <- class_to_color(subset_to_class(df$subset))




# Create the initial plot object.
pl <- ggplot(df, aes(x=c(x,xend), y=c(y,yend)))

# Removing edges completely from the figure if they don't represent a high enough probability.
df <- df[df$value > weight_threshold,]




# MOVED THIS UP TO MAKE SURE THE THINGS THAT DON'T HAVE CONNECTIONS STILL GET GPLOOTED
# Create the initial plot object.
#pl <- ggplot(df, aes(x=c(x,xend), y=c(y,yend)))

# Remembering where each factor for the subsets and topics should map to on the continuous y-axis (both left and right).
numbers <- unique(df$y)
left_hashmap<- hashmap(df$y, as.character(df$label_left))
right_hashmap <- hashmap(df$yend, as.character(df$label_right))
num_to_left_label <- function(x){return(left_hashmap[[x]])}
num_to_right_label <- function(x){return(right_hashmap[[x]])}



# Adding each line to plot from the table of lines one-at-a-time in this for loop. Not sure if there's a more elegant way to do this with ggplot.
for (i in 1:nrow(df)){
  pl <- pl + geom_segment(aes_string(x=df[i,"x"], y=df[i,"y"], xend=df[i,"xend"], yend=df[i,"yend"]), "size"=df[i,"size"], "alpha"=df[i,"alpha"], "color"=df[i, "color"])
}


# Create the rest of plot's look. The sec.axis section specifies what the y-axis on the right side looks like.
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
        axis.ticks.x = element_blank(),
        axis.text.y = element_text(hjust=0))
  
  

# Saving the plot to a .png file.
path <- "/Users/irbraun/Desktop/matching.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=30, height=20, units=c("cm"), dpi=300, limitsize=FALSE)



