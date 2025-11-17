library(ggplot2)
library(dplyr)

args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
  stop("No file provided", call.=FALSE)
}

input_file <- args[1]

df <- read.table(file = input_file, header=T)
df <- df[df$Label != "seq_vendor",]
df$Label <- as.factor(df$Label)
colors <- as.numeric(df$Label)
plot(x = df$N, y = df$GFlops, xlab = "Taille de la matrice (M=N=K)", ylab="Performance (GFlop/s)", log="x", col=colors)
legend("topleft",legend = levels(df$Label),col = 1:length(levels(df$Label)),pch = 19)

df <- df %>%
  group_by(Label, N) %>%
  summarize(
    GFlops_mean = mean(GFlops),
    GFlops_min = min(GFlops),
    GFlops_max = max(GFlops),
    .groups = 'drop'
  )

plot <- ggplot(data = df, mapping = aes(x = N, y = GFlops_mean, colour = Label)) +
  geom_line() +
  geom_point(aes(y=GFlops_mean))+
  scale_x_continuous(trans='log2') +
  scale_y_continuous(trans='log2') +
  labs(x="Matrix size (N=M=K)", y="Performance (GFlop/s)")

ggsave(paste(input_file, "_simple.png", sep=""), plot = plot, width = 10, height = 6, dpi = 300)
