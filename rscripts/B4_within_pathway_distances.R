
library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(lattice)
library(hashmap)



df <- read.csv(file="/Users/irbraun/Desktop/droplet/phenologs-with-oats/outputs/stacked_10_19_2020_h11m09s50_4661_balanced/stacked_pmn_only_within_distances.csv")

df <- df %>% filter(n>1)


names_path <- "/Users/irbraun/phenologs-with-oats/names.tsv"


# Only look at the top 50 pathways, if more were found.
df <- df[1:min(50,nrow(df)),]



# Include the value for n with the full names.
df$name_for_plot = paste(df$full_name, " (n=", df$n, ")", sep="") 









# THESE CHANGES SHOULD BE DONE IN TEH NOTEBOOK, GET RID OF UNUSED COLUMN, JUST NEED THE ONE NAME ONE.
# NEW THING CHECK THIS AND MOVE TO OTHER FIGURE SCRIPTS AS WELL.
# This should be done in the notebook!!!!! just have it write the full name with hyperparams instead.

df_long <- gather(df, approach, avg_percentile, -group_id, -full_name, -name_for_plot, -n)
head(df_long)



names_df <- read.csv(file=names_path, header=T, sep="\t")

df_long <- merge(x=df_long, y=names_df, by.x=c("approach"), by.y=c("name_in_notebook"), all.x=TRUE)

# Were there any approaches that didn't map in the naming file?
df_long[rowSums(is.na(df_long)) > 0,]

# Get rid of them.
df_long <- df_long %>% drop_na()


df_long <- df_long %>% filter(!class %in% c("Curation","Baseline"))


head(df_long)





# Transform the dataframe by collapsing to method type, and remembering the average, min, and max metrics obtained by each class of method.
df_long_t <- data.frame(df_long %>% group_by(group_id,full_name,n,name_for_plot,class) %>% summarize(avg=mean(avg_percentile)))
head(df_long_t)



#df_long$old <- paste(df_long$method, df$hyperparameters, sep="__")
#df_long <- subset(df_long, select = -c(order, method, hyperparameters, group) )






#df_long <- gather(df, approach, avg_percentile, -group_id, -full_name, -name_for_plot, -n, -mean_avg_pair_percentile, -mean_group_rank)
#head(df_long)


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






ggplot(df_long_t, aes(x=avg, y=reorder(name_for_plot,-avg), fill=class)) + 
  scale_fill_manual(name="Shared", values=color_mapping) +
  #geom_point(colour="black", fill="black", pch=21, size=3) +
  geom_point(colour="black", pch=21, size=3, alpha=0.8) +
  theme_bw() +
  ylab("Biochemical Pathway") +
  xlab("Mean Within-Pathway Distance Percentile") 


path <- "~/Desktop/plot.png"
ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=30, height=18, units=c("cm"), dpi=400, limitsize=FALSE)









# 
# ggplot(df_long, aes(x=avg_percentile, y=reorder(name_for_plot,-mean_avg_pair_percentile), fill=name, alpha=0.3)) + 
#   #geom_point(colour="black", fill="black", pch=21, size=3) +
#   geom_point(colour="black", pch=21, size=3) +
#   theme_bw() +
#   ylab("Biochemical Pathway") +
#   xlab("Mean Within-Pathway Distance Percentile") 
# 
# 
# path <- "~/Desktop/plot.png"
# ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=20, height=12, units=c("cm"), dpi=500, limitsize=FALSE)



