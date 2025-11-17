library(dplyr)
library(ggplot2)
library(tidyr)


args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
  stop("No file provided", call.=FALSE)
}

input_file <- args[1]

df <- read.table(file = input_file, header = TRUE)

reference_label <- "vendor"

df_avg <- df %>%
  unite("Matrice_Size", M, N, K, sep = "x", remove = FALSE) %>%
  group_by(Matrice_Size, Label) %>%
  summarise(Mean_GFlops = mean(GFlops), .groups = 'drop')

df_reference <- df_avg %>%
  filter(Label == reference_label) %>%
  select(Matrice_Size, Reference_GFlops = Mean_GFlops)

df_normalized <- df_avg %>%
  left_join(df_reference, by = "Matrice_Size") %>%
  mutate(Normalized_GFlops = Mean_GFlops / Reference_GFlops) %>%
  mutate(Matrice_Size = as.factor(Matrice_Size))

df_plot <- df_normalized %>%
  filter(Label != reference_label)

plot <- ggplot(df_plot, aes(x = Matrice_Size, y = Normalized_GFlops, fill = Label)) +
  
  geom_bar(stat = "identity", position = position_dodge()) +
  
  geom_hline(
    yintercept = 1.0,
    linetype = "dashed",
    color = "red",
    linewidth = 1
  ) +
  
  geom_text(
    aes(x = Inf, y = 1.0, label = paste(reference_label, "(1.0)"), hjust = 1.1, vjust = -0.5),
    color = "red",
    check_overlap = TRUE
  ) +
  
  labs(
    x = "Matrice Size (MxNxK)",
    y = "Normalized GFlop/s (Vendor = 1.0)",
    fill = "Implementation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
  panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = "white", colour = NA))
  

ggsave(paste(input_file, "_bar.png", sep=""), plot = plot, width = 10, height = 6, dpi = 300)
