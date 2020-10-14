library(ggplot2)
library(tidyr)
library(dplyr)
library(hashmap)




# Reading in the topic modeling file file from the notebook output.
df <- read.csv(file="/Users/irbraun/phenologs-with-oats/outputs/05_18_2020_h11m47s26/part_6_topic_modeling.csv")




# Adjusting the y-values a little bit for the start and end of each line to make distinctions between classes clear.
left_x_value <- 0
right_x_value <- 10

# Compressing the y-axis spacing on the left side.
left_multiplier <- 0.8
df[df$x==left_x_value,]$y <- df[df$x==left_x_value,]$y * left_multiplier     

# Get the classes in the order they'll appear (bottom to top, as y-axis value increase).
# Note that right now this just assumes they'll be in the correct order in the file.
# In reality this should really sort by y value after removing duplicates.
classes_df <- df[!duplicated(df$class_str),]
classes_df$height_addition <- seq(0, nrow(classes_df)-1)
class_to_addition_hashmap<- hashmap(as.character(classes_df$class_str), classes_df$height_addition)
class_to_addition <- function(x){return(class_to_addition_hashmap[[x]])}
df[df$x==left_x_value,]$y <- df[df$x==left_x_value,]$y + class_to_addition(df[df$x==left_x_value,]$class_str)

# Once the left side is done, make the right side match it's height but be evenly spaced.
linear_conversion <- function(x, new_min, new_max, old_min, old_max){return((((x - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min)}
min_left <- min(df[df$x==left_x_value,]$y)
max_left <- max(df[df$x==left_x_value,]$y)
min_right <- min(df[df$x==right_x_value,]$y)
max_right <- max(df[df$x==right_x_value,]$y)
df[df$x==right_x_value,]$y <- linear_conversion(df[df$x==right_x_value,]$y, min_left, max_left, min_right, max_right)

# Replacing pipes with commas and fixing the spacing in the lists of tokens for each topic.
df$topic_str <- gsub('\\|', ', ', df$topic_str)










# Re-scaling the alpha and size parameters for each line as linear functions of the weight ranges.
min_alpha <- 0.01
max_alpha <- 1.0
min_size <- 0.01
max_size <- 1
min_weight <- min(df$weight)
max_weight <- max(df$weight)
df$alpha <- linear_conversion(df$weight, min_alpha, max_alpha, min_weight, max_weight)
df$size <- linear_conversion(df$weight, min_size, max_size, min_weight, max_weight)



# Setting up the subset and topic labels for the plot.
left_numbers <- unique(df[df$x==left_x_value,]$y)
right_numbers <- unique(df[df$x==right_x_value,]$y)
left_hashmap<- hashmap(df[df$x==left_x_value,]$y, as.character(df[df$x==left_x_value,]$subset_str))
right_hashmap<- hashmap(df[df$x==right_x_value,]$y, as.character(df[df$x==right_x_value,]$topic_str))
num_to_left_label <- function(x){return(left_hashmap[[x]])}
num_to_right_label <- function(x){return(right_hashmap[[x]])}




# Making the actual plot.
ggplot(df, aes(x=x, y=y, group=line_number, alpha=alpha, size=size)) + geom_line(color="black") + scale_size_identity() +
  theme_bw() +
  scale_y_continuous(name="", breaks=left_numbers, labels=num_to_left_label, expand=c(0.01, 0.01),
                     sec.axis = sec_axis(~. , name="", breaks=right_numbers, labels=num_to_right_label)) +
  scale_x_continuous(breaks=seq(0,10), limits=c(0,10), expand=c(0.01, 0.01)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_text(hjust=0),
        legend.position = "none")




# Saving the plot to a file.
path <- "/Users/irbraun/Desktop/matching.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=30, height=20, units=c("cm"), dpi=300, limitsize=FALSE)


  