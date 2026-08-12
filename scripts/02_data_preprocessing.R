# Q2_data_preprocessing

# Check missing values before recoding
missing_summary = data.frame(
Variable = names(WD),
Missing_Count = sapply(WD, function(x) sum(is.na(x))),
Missing_Percentage = round(sapply(WD, function(x) mean(is.na(x)) * 100), 2)
)
# Filter to variables with missing values only
missing_summary_nonzero = missing_summary[
missing_summary$Missing_Count > 0,
]
missing_summary_nonzero
# Check negative-coded responses before cleaning
negative_summary = data.frame(
Variable = names(WD),
Negative_Count = sapply(WD, function(x) {
if (is.numeric(x)) {
sum(x < 0, na.rm = TRUE)
} else {
0
}
}),
Negative_Percentage = round(sapply(WD, function(x) {
if (is.numeric(x)) {
mean(x < 0, na.rm = TRUE) * 100
} else {
0
}
}), 2)
)
# Filter to variables with negative-coded responses only
negative_summary_nonzero = negative_summary[
negative_summary$Negative_Count > 0,
]
negative_summary_nonzero
# Figure 6: Plot negative-coded response percentages per variable
plot_negative = ggplot(
negative_summary_nonzero,
aes(x = reorder(Variable, Negative_Percentage), y = Negative_Percentage)
) +
geom_bar(stat = "identity", fill = "salmon", colour = "black") +
coord_flip() +
labs(
title = "Negative-coded Responses Before Pre-processing",
x = "Variable",
y = "Negative-coded Response Percentage"
) +
theme_minimal() +
theme(
plot.title = element_text(size = 14, face = "bold"),
axis.title.x = element_text(size = 11, face = "bold"),
axis.title.y = element_text(size = 11, face = "bold"),
axis.text.x = element_text(size = 9, face = "bold"),
axis.text.y = element_text(size = 8, face = "bold")
)
plot_negative
# Save plot
ggsave(
filename = "Q2_negative_values_before_preprocessing.png",
plot = plot_negative,
width = 10,
height = 7,
units = "in",
dpi = 300
)
# Recode negative values in predictor variables to NA
WD_clean = WD
WD_clean[predictor_vars] = lapply(WD_clean[predictor_vars], function(x) {
x[x < 0] = NA
return(x)
})
# Check missing values after recoding
missing_after_recode = data.frame(
Variable = names(WD_clean),
Missing_Count = sapply(WD_clean, function(x) sum(is.na(x))),
Missing_Percentage = round(sapply(WD_clean, function(x) mean(is.na(x)) * 100), 2)
)
# Filter to variables with missing values only
missing_after_recode_nonzero = missing_after_recode[
missing_after_recode$Missing_Count > 0,
]
missing_after_recode_nonzero
# Apply complete-case filtering to create final modelling dataset
WD_model = WD_clean[complete.cases(WD_clean), ]
# Check dimensions before and after preprocessing
dim(WD)
dim(WD_model)
# Confirm no missing values remain
sum(is.na(WD_model))
# Convert class variables to factors
WD_model[class_vars] = lapply(WD_model[class_vars], factor)
# Convert predictor variables to factors
WD_model[predictor_vars] = lapply(WD_model[predictor_vars], factor)
# Confirm factor conversion
str(WD_model[class_vars])
str(WD_model[predictor_vars])
# Check class balance after preprocessing
class_balance_model = data.frame()
for (target in class_vars) {
counts = table(WD_model[[target]])
proportions = prop.table(counts)
temp = data.frame(
ClassVariable = target,
ClassValue = names(counts),
Count = as.numeric(counts),
Proportion = round(as.numeric(proportions), 4),
Percentage = round(as.numeric(proportions) * 100, 2)
)
class_balance_model = rbind(class_balance_model, temp)
}
class_balance_model
# Figure 7: Plot class balance after preprocessing
plot_class_balance_model = ggplot(
class_balance_model,
aes(x = ClassVariable, y = Percentage, fill = ClassValue)
) +
geom_bar(stat = "identity", position = "dodge", colour = "black") +
labs(
title = "Class Balance After Pre-processing",
x = "Class Variable",
y = "Percentage of Modelling Data",
fill = "Class Value"
) +
theme_minimal() +
theme(
plot.title = element_text(size = 14, face = "bold"),
axis.title.x = element_text(size = 11, face = "bold"),
axis.title.y = element_text(size = 11, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold"),
axis.text.y = element_text(size = 9, face = "bold"),
legend.title = element_text(face = "bold"),
legend.text = element_text(face = "bold")
)
plot_class_balance_model
# Save plot
ggsave(
filename = "Q2_class_balance_after_preprocessing.png",
plot = plot_class_balance_model,
width = 9,
height = 6,
units = "in",
dpi = 300
)
