library(rjson)
library(dplyr)






# Read in the file that contains the all the precision and recall values for each method.
df <- read.csv(file="/Users/irbraun/Desktop/test.csv")
head(df)


ggplot(df, aes(x=bin_center, y=frequency, fill=approach)) +
  theme_bw() +
  #geom_smooth()
  #geom_line()
  geom_bar(stat="identity", position="identity", width=0.01, alpha=0.4)





df_p = data.frame(result$positives)
df_n = data.frame(result$negatives,)



df <- dplyr::bind_rows(data.frame(result$positives), data.frame(result$negatives), .id = "source")






df <- data.frame(result)
head(df)

df <- gather(df, kind, value, positives, negatives)
head(df)



# Generate the plot of sentence number distribution and save to a file.
ggplot(df, aes(x=num_sents)) +
  geom_histogram(alpha=0.9, col="black", fill="lightgray", bins=30) +
  theme_bw() +
  facet_grid(rows=vars(species),cols=vars(),scale="free") +
  scale_x_continuous(breaks=seq(0,sent_limit,10), limits=c(0,sent_limit+2), expand = c(0.01, 0)) +
  theme(plot.title = element_text(lineheight=1.0, face="bold", hjust=0.5), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.direction = "vertical",
        legend.position = "right")+ 
  ylab("Frequency") +
  xlab("Number of Sentences")




ggplot(a, aes(x=values)) + geom_density()