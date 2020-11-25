library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)
library(RColorBrewer)
library(hashmap)
library(ggrepel)



# The input and output files that this script uses and creates.
input_path <- "/Users/irbraun/phenologs-with-oats/outputs/composition_11_22_2020_h12m47s02_5791/token_frequencies.csv"
output_path <- "/Users/irbraun/phenologs-with-oats/figs/vocabs_compared.png"



df <- read.csv(file=input_path, header=T, sep=",")
df$token <- as.character(df$token)


# Thresholds and assigning labels to certain points.
threshold_zma <- 0.75
threshold_ath <- 0.75
df$plotword <- ""
df[df$zma_rate>=threshold_zma,]$plotword <- df[df$zma_rate>=threshold_zma,]$token
df[df$ath_rate>=threshold_ath,]$plotword <- df[df$ath_rate>=threshold_ath,]$token


df$exclusive <- "neither"
df[df$zma_freq == 0,]$exclusive <- "all_ath"
df[df$ath_freq == 0,]$exclusive <- "all_zma"


# Pick from those colors to match the same number of values present in group_names.
group_colors <- c("#FFDA67", "#E69238", "#000000")
group_names <-  c("all_ath","all_zma","neither")
group_mapping <- setNames(group_colors, group_names)

ggplot(df, aes(x=zma_rate, y=ath_rate)) +
  geom_point(alpha=0.6) +
  geom_text_repel(aes(label=plotword),color="black", hjust=-0.1, vjust=0.3) +
  theme_bw() +
  scale_color_manual(name="Only",values=group_mapping) +
  scale_x_continuous(breaks=seq(0,3.0,0.5), limits=c(0,3), expand = c(0.01, 0)) +
  scale_y_continuous(breaks=seq(0,2.0,0.5), limits=c(0,2), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), 
        #legend.direction = "vertical",
        #legend.position = "right")+ 
        legend.position = "none")+
  ylab("Arabidopsis (Per 100 Words)") +
  xlab("Maize (Per 100 Words)")



# Save the image of the plot.
ggsave(output_path, plot=last_plot(), device="png", path=NULL, scale=1, width=20, height=14, units=c("cm"), dpi=300, limitsize=TRUE)









# # Pick from those colors to match the same number of values present in group_names.
# group_colors <- c("#575757", "#E7E8EA")
# group_names <-  c(TRUE,FALSE)
# group_mapping <- setNames(group_colors, group_names)
# label_mapping <- setNames(group_names, c("yesss", "no"))
# 
# 
# 
# # What if we want the frequencies by which range they fit into?
# # Have to make the numeric column categorical by binning.
# bin_breaks <- c(-Inf, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, Inf)
# bin_names <- c("1","2","3","4","5","6", "7", "8", "9", "10", "11-100", ">100")
# df$ath_freq_bin <- cut(df$ath_freq, breaks=bin_breaks, labels=bin_names)
# df$zma_freq_bin <- cut(df$zma_freq, breaks=bin_breaks, labels=bin_names)
# df$ath_present <- (df$ath_freq>0)
# df$zma_present <- (df$zma_freq>0)
# 
# 
# 
# 
# # Make a new dataframe in the shape necessary for making a faceted plot easily.
# a = df[df["ath_freq"]>0,c("ath_freq_bin","zma_present")]
# a$facet <- "Arabidopsis"
# colnames(a) <- c("freq", "other", "facet")
# b = df[df["zma_freq"]>0,c("zma_freq_bin","ath_present")]
# b$facet <- "Maize"
# colnames(b) <- c("freq", "other", "facet")
# facet_df <- rbind(a,b)
# head(facet_df)
# 
# # Size of the plot in cm when saving to file.
# h = 10
# w = 14
# 
# # Generate the plot for the frequency of lemmes in Arabidopsis phenotypes.
# ggplot(facet_df, aes(x=freq, fill=other)) +
#   geom_histogram(alpha=1.0, stat="count") + 
#   geom_histogram(alpha=1.0, stat="count", color="black") +
#   facet_grid(rows=vars(facet), scales="free") +
#   scale_fill_manual(name="Frequecy>0 in other Species", values=group_mapping, labels=c("No","Yes")) +
#   theme_bw() +
#   theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
#         panel.grid.major = element_blank(), 
#         legend.direction = "horizontal",
#         legend.position = "bottom")+ 
#   ylab("Bin Size") +
#   xlab("Words Binned by Count")
# 
# # Save the image of the plot.
# path <- "~/Desktop/facet_plot.png"
# ggsave(path, plot=last_plot(), device="png", path=NULL, scale=1, width=w, height=h, units=c("cm"), dpi=300, limitsize=TRUE)



# What about a simple Venn Diagram for the vocabularies too?
ath <- nrow(df[df["ath_freq"]>0,])
overlap <- nrow(df[(df["ath_freq"]>0) & (df["zma_freq"]>0),])
zma <- nrow(df[df["zma_freq"]>0,])
ath_only <- ath-overlap
zma_only <- zma-overlap
paste(ath_only, overlap, zma_only)







