library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)




# Reading in the CSV file of results.
PATH <- "/Users/irbraun/phenologs-with-oats/outputs/03_02_2020_h12m06s11/part_6_full_table.csv"
PATH <- "/Users/irbraun/Desktop/mgc_plots/s3.csv"
#PATH <- "/Users/irbraun/Desktop/mgc_plots/pw3.csv"
df <- read.csv(file=PATH, header=T, sep=",")




# Subset the dataframe based on a particular topic or dataset if necesssary.
#df <- subset(df, Topic=="Biochemical Pathways" & Data=="Entire Dataset")
#df <- subset(df, Topic=="Biochemical Pathways")


# Collapse the dataframe to average across all hyperparameters for each method and find the standard deviation.
map <- hashmap(as.character(df$Method), as.character(df$Group))
mapping_function <- function(x){return(map[[x]])}
df <- data.frame(df %>% group_by(Method) %>% summarize(f1_max_avg=mean(f1_max), f1_max_sd=mean(f1_std), Order=mean(Order)))
#df[is.na(df)] <- 0.000 # Replace NA with 0 so that sd bars of height 0 can be added to bars for methods with no hyperparamter variation.
df$Category <- mapping_function(df$Method)



# Colors used.
#581845 (purple)
#900C3F (dark red)
#C70039 (red)
#FF5733 (orange)
#FFC300 (yellow)
#DAF7A6 (light green)


# Gokul
#7571B3 blue
#DA6000 orange
#189E78 green



#https://meyerweb.com/eric/tools/color-blend/#581845:900C3F::hex
##771142 (halfway between purple and dark red)




# Organizing the different methods presented into general categories.
df$group <- factor(df$Category, levels = c("NLP","Embedding","Curated","Other"))
group_colors_bw <- c(grey.colors(n=3,start=0.1,end=1.0,alpha=1))
group_colors <- c("#581845", "#900C3F", "#C70039", "#FF5733", "#FFC300", "#DAF7A6", "#771142")
group_colors <- c("#7571B3", "#DA6000", "#189E78", "#000000")

# Pick from those colors to match the same number of values present in group_names.
group_colors <- c(group_colors[2], group_colors[3], group_colors[4], group_colors[1])
group_names <-  c("NLP","Embedding","Curated", "Other")
group_mapping <- setNames(group_colors, group_names)




# Change these to change what values are used for plotting, so that the ggplot() call doesn't need to be changed.
baseline = read.csv(file=PATH, header=T, sep=",")$baseline[1]
y_lim <- 0.75
step_size <- 0.15
df$metric_to_use <- df$f1_max_avg
df$error_to_use <- df$f1_max_sd




ggplot(data=df, aes(x=reorder(Method,Order),y=metric_to_use,fill=group))+geom_bar(stat="identity") + geom_bar(stat="identity", color="black") + geom_errorbar(aes(ymin=metric_to_use, ymax=metric_to_use+error_to_use), width=.3) +
  #facet_grid(cols=vars(Data), rows=vars(Topic), scales="free") +
  theme_bw() +
  scale_fill_manual(name="Approach Used",values=group_mapping) +
  geom_abline(slope=0, intercept=baseline,  col = "gray", lty=2) +
  scale_x_discrete(breaks=df$Method,labels=df$Method) +
  scale_y_continuous(breaks=seq(0,y_lim,step_size), limits=c(NA,y_lim), expand = c(0, 0)) +
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
  ylab("Performance (F1)")


# Save the image of the plot.
path <- "~/Desktop/mgc_plots/ps3.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=7, height=5.5, units=c("cm"), dpi=300, limitsize=TRUE)


# 7 by 3.5 for without x axis names
# 7 by 5.5 for with x axis names

