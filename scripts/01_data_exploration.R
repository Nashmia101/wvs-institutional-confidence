# Q1_data_exploration

# Clear workspace and set random seed using student ID
rm(list = ls())
set.seed(34091904)
# Load the full WVS binary dataset
WD = read.csv("WVSBinaryExtract.csv")
# Randomly select 30 predictor columns (cols 3-49) and 3 binary class variables (cols 50-63)
selected_cols = c(sample(3:49, 30), sample(50:63, 3))
# Subset dataset to include Country, Wave, and selected columns
WD = WD[c(1:2, selected_cols)]
# Randomly sample 20,000 observations without replacement
WD = WD[sample(nrow(WD), 20000, replace = FALSE),]
# Check dimensions and column names of the individual dataset
dim(WD)
names(WD)
# Identify the 3 binary class variables (columns 33-35)
class_vars = names(WD)[33:35]
# Identify the 30 predictor variables (columns 3-32)
predictor_vars = names(WD)[3:32]
# Confirm variable names
class_vars
predictor_vars
# Compute class distribution including missing values to assess missingness
class_distribution = data.frame()
for (target in class_vars) {
counts = table(WD[[target]], useNA = "ifany")
proportions = prop.table(counts)
temp = data.frame(
ClassVariable = target,
ClassValue = names(counts),
Count = as.numeric(counts),
Proportion = round(as.numeric(proportions), 4),
Percentage = round(as.numeric(proportions) * 100, 2)
)
class_distribution = rbind(class_distribution, temp)
}
class_distribution
# Compute class distribution on valid responses only (excluding NAs)
class_distribution_valid = data.frame()
for (target in class_vars) {
valid_values = WD[[target]][!is.na(WD[[target]])]
counts = table(valid_values)
proportions = prop.table(counts)
temp = data.frame(
ClassVariable = target,
ClassValue = names(counts),
Count = as.numeric(counts),
Proportion = round(as.numeric(proportions), 4),
Percentage = round(as.numeric(proportions) * 100, 2)
)
class_distribution_valid = rbind(class_distribution_valid, temp)
}
class_distribution_valid
# Figure 2: Grouped bar chart comparing High (1) and Low (0) proportions across class variables
ggplot(class_distribution_valid,
aes(x = ClassVariable, y = Percentage, fill = ClassValue)) +
geom_bar(stat = "identity", position = "dodge") +
labs(
title = "Distribution of High and Low Confidence Classes",
x = "Class Variable",
y = "Percentage of Valid Responses",
fill = "Class Value"
) +
theme_minimal()
# Figure 3(c): Reshape data to long format and plot predictor distributions by CCivilService
boxplot_data_civil = WD %>%
select(CCivilService, all_of(predictor_vars)) %>%
filter(!is.na(CCivilService)) %>%
pivot_longer(
cols = all_of(predictor_vars),
names_to = "Predictor",
values_to = "Value"
)
plot_civil = ggplot(
boxplot_data_civil,
aes(x = factor(CCivilService), y = Value, fill = factor(CCivilService))
) +
geom_boxplot(outlier.size = 0.25) +
facet_wrap(~ Predictor, scales = "free_y", ncol = 6) +
labs(
title = "Distribution of Predictor Variables by CCivilService",
x = "Class Value",
y = "Predictor Value",
fill = "Class"
) +
theme_minimal() +
theme(
strip.text = element_text(size = 10, face = "bold"),
axis.title.x = element_text(size = 10, face = "bold"),
axis.title.y = element_text(size = 10, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold"),
axis.text.y = element_text(size = 10, face = "bold"),
plot.title = element_text(size = 12, face = "bold"),
legend.position = "bottom",
legend.title = element_text(face = "bold"),
legend.text = element_text(face = "bold")
)
#saving image
ggsave(
filename = "Q1_boxplots_CCivilService.png",
plot = plot_civil,
width = 16,
height = 12,
units = "in",
dpi = 300
)
# Figure 3(b): Reshape data to long format and plot predictor distributions by CChurches
boxplot_data_church = WD %>%
select(CChurches, all_of(predictor_vars)) %>%
filter(!is.na(CChurches)) %>%
pivot_longer(
cols = all_of(predictor_vars),
names_to = "Predictor",
values_to = "Value"
)
plot_church = ggplot(
boxplot_data_church,
aes(x = factor(CChurches), y = Value, fill = factor(CChurches))
) +
geom_boxplot(outlier.size = 0.25) +
facet_wrap(~ Predictor, scales = "free_y", ncol = 6) +
labs(
title = "Distribution of Predictor Variables by CChurches",
x = "Class Value",
y = "Predictor Value",
fill = "Class"
) +
theme_minimal() +
theme(
strip.text = element_text(size = 10, face = "bold"),
axis.title.x = element_text(size = 10, face = "bold"),
axis.title.y = element_text(size = 10, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold"),
axis.text.y = element_text(size = 10, face = "bold"),
plot.title = element_text(size = 12, face = "bold"),
legend.position = "bottom",
legend.title = element_text(face = "bold"),
legend.text = element_text(face = "bold")
)
# saving image
ggsave(
filename = "Q1_boxplots_CChurches.png",
plot = plot_church,
width = 16,
height = 12,
units = "in",
dpi = 300
)
# Figure 3(a): Reshape data to long format and plot predictor distributions by CArmedForces
boxplot_data_army = WD %>%
select(CArmedForces, all_of(predictor_vars)) %>%
filter(!is.na(CArmedForces)) %>%
pivot_longer(
cols = all_of(predictor_vars),
names_to = "Predictor",
values_to = "Value"
)
plot_army = ggplot(
boxplot_data_army,
aes(x = factor(CArmedForces), y = Value, fill = factor(CArmedForces))
) +
geom_boxplot(outlier.size = 0.25) +
facet_wrap(~ Predictor, scales = "free_y", ncol = 6) +
labs(
title = "Distribution of Predictor Variables by CArmedForces",
x = "Class Value",
y = "Predictor Value",
fill = "Class"
) +
theme_minimal() +
theme(
strip.text = element_text(size = 10, face = "bold"),
axis.title.x = element_text(size = 10, face = "bold"),
axis.title.y = element_text(size = 10, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold"),
axis.text.y = element_text(size = 10, face = "bold"),
plot.title = element_text(size = 12, face = "bold"),
legend.position = "bottom",
legend.title = element_text(face = "bold"),
legend.text = element_text(face = "bold")
)
#saving image
ggsave(
filename = "Q1_boxplots_CArmedForces.png",
plot = plot_army,
width = 16,
height = 12,
units = "in",
dpi = 300
)
# Figure 4: Compute and plot correlation matrix for all predictor variables
cor_matrix = cor(
WD[predictor_vars],
use = "pairwise.complete.obs"
)
# saving image
png(
filename = "Q1_correlation_matrix_with_numbers.png",
width = 3200,
height = 3200,
res = 300
)
corrplot(
cor_matrix,
method = "color",
type = "upper",
addCoef.col = "black",
number.cex = 0.45,
tl.cex = 0.55,
tl.col = "black",
tl.srt = 45,
col = colorRampPalette(c("red", "white", "blue"))(200),
title = "Correlation Matrix of Predictor Variables",
mar = c(0, 0, 2, 0)
)
dev.off()
# Figure 1: Plot overall distribution of all predictor variables using faceted boxplots
overall_boxplot_data = WD %>%
select(all_of(predictor_vars)) %>%
pivot_longer(
cols = everything(),
names_to = "Predictor",
values_to = "Value"
)
plot_overall_box_facet = ggplot(
overall_boxplot_data,
aes(x = "", y = Value)
) +
geom_boxplot(fill = "skyblue", outlier.size = 0.3) +
facet_wrap(~ Predictor, scales = "free_y", ncol = 6) +
labs(
title = "Overall Distribution of Predictor Variables",
x = "",
y = "Value"
) +
theme_minimal() +
theme(
strip.text = element_text(size = 10, face = "bold"),
axis.title.x = element_text(size = 10, face = "bold"),
axis.title.y = element_text(size = 10, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold"),
axis.text.y = element_text(size = 10, face = "bold"),
plot.title = element_text(size = 12, face = "bold"),
legend.position = "bottom",
legend.title = element_text(face = "bold")
)
plot_overall_box_facet
# saving image
ggsave(
filename = "Q1_overall_predictor_boxplot_facet.png",
plot = plot_overall_box_facet,
width = 16,
height = 12,
units = "in",
dpi = 300
)
