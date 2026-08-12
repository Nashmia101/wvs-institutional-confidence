# Q12_ann_wave_comparison

# Use the ZAF test data from Question 11 and add a row ID
test_country_q12 = test_country_q11 %>%
mutate(Test_Row_ID = row_number())
# Keep only complete rows that match the rows used by the Q11 ANN model
usable_test_country_q12 = test_country_q12 %>%
filter(
complete.cases(
select(
.,
all_of(c(all_predictors_q11, target_q11))
)
)
)
# Check that usable test rows match the ANN outputs from Question 11
nrow(usable_test_country_q12)
length(ann_all_q11$actual)
length(ann_all_q11$probabilities)
length(ann_all_q11$predictions)
# Count the number of usable ZAF test observations in each wave
wave_counts_q12 = usable_test_country_q12 %>%
group_by(Wave) %>%
summarise(
Number_Observations = n(),
.groups = "drop"
) %>%
arrange(desc(Number_Observations))
wave_counts_q12
# Select the two waves with the greatest number of observations
top_two_waves_q12 = wave_counts_q12$Wave[1:2]
top_two_waves_q12
# Create a bar chart showing the number of observations in each selected wave
wave_counts_plot_q12 = wave_counts_q12 %>%
mutate(
Wave = factor(Wave, levels = Wave)
)
ggplot(
wave_counts_plot_q12,
aes(x = Wave, y = Number_Observations, fill = Wave)
) +
geom_col(width = 0.6) +
geom_text(
aes(label = Number_Observations),
vjust = -0.4,
size = 5,
fontface = "bold"
) +
labs(
title = "Number of ZAF Test Observations by Wave",
subtitle = "Waves 6 and 3 had the greatest number of observations",
x = "Wave",
y = "Number of Test Observations"
) +
ylim(0, max(wave_counts_plot_q12$Number_Observations) + 8) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
plot.subtitle = element_text(size = 12, hjust = 0.5),
axis.title.x = element_text(size = 14, face = "bold"),
axis.title.y = element_text(size = 14, face = "bold"),
axis.text.x = element_text(size = 12, face = "bold"),
axis.text.y = element_text(size = 12, face = "bold"),
legend.position = "none"
)
# Attach the Q11 ANN actual values, predictions and probabilities to the ZAF test rows
q12_test_predictions = usable_test_country_q12 %>%
mutate(
Actual = ann_all_q11$actual,
Predicted = ann_all_q11$predictions,
Probability_High_Confidence = ann_all_q11$probabilities
)
# Keep only the two selected waves for comparison
q12_top_wave_predictions = q12_test_predictions %>%
filter(Wave %in% top_two_waves_q12)
# Summarise class distribution in the selected waves
q12_wave_class_summary = q12_top_wave_predictions %>%
group_by(Wave) %>%
summarise(
Number_Observations = n(),
Low_Confidence_Actual = sum(Actual == "0"),
High_Confidence_Actual = sum(Actual == "1"),
.groups = "drop"
)
q12_wave_class_summary
# Calculate ANN performance separately for each selected wave
q12_wave_results = data.frame()
q12_wave_confusion_matrices = list()
q12_wave_roc_objects = list()
for (current_wave in top_two_waves_q12) {
# Filter the test predictions to the current wave
wave_data = q12_top_wave_predictions %>%
filter(Wave == current_wave)
# Store actual labels, predicted labels and predicted probabilities
wave_actual = factor(
wave_data$Actual,
levels = c("0", "1")
)
wave_predicted = factor(
wave_data$Predicted,
levels = c("0", "1")
)
wave_probabilities = wave_data$Probability_High_Confidence
# Create confusion matrix for the current wave
wave_cm = table(
Actual = wave_actual,
Predicted = wave_predicted
)
q12_wave_confusion_matrices[[as.character(current_wave)]] = wave_cm
# Calculate accuracy, precision, recall and F1-score
wave_metrics = calculate_ann_metrics_q11(
actual = wave_actual,
predicted = wave_predicted
)
# Calculate ROC and AUC using predicted probabilities
wave_roc = roc(
response = wave_actual,
predictor = wave_probabilities,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
wave_auc = round(
as.numeric(auc(wave_roc)),
4
)
q12_wave_roc_objects[[as.character(current_wave)]] = wave_roc
# Store wave-level model performance
q12_wave_results = rbind(
q12_wave_results,
data.frame(
Wave = current_wave,
Number_Observations = nrow(wave_data),
Accuracy = wave_metrics$Accuracy,
Precision = wave_metrics$Precision,
Recall = wave_metrics$Recall,
F1_Score = wave_metrics$F1_Score,
AUC = wave_auc
)
)
}
q12_wave_results
q12_wave_confusion_matrices
# Create final performance table for the two selected waves
q12_final_table = q12_wave_results %>%
arrange(Wave) %>%
mutate(
Accuracy = round(Accuracy, 4),
Precision = round(Precision, 4),
Recall = round(Recall, 4),
F1_Score = round(F1_Score, 4),
AUC = round(AUC, 4)
)
# Prepare data for grouped bar chart of performance metrics
q12_wave_results_long = q12_wave_results %>%
select(Wave, Accuracy, Precision, Recall, F1_Score, AUC) %>%
pivot_longer(
cols = c(Accuracy, Precision, Recall, F1_Score, AUC),
names_to = "Metric",
values_to = "Value"
) %>%
mutate(
Wave = factor(Wave),
Metric = factor(
Metric,
levels = c("Accuracy", "Precision", "Recall", "F1_Score", "AUC")
)
)
# Plot performance metrics for Wave 3 and Wave 6
ggplot(
q12_wave_results_long,
aes(x = Metric, y = Value, fill = Wave)
) +
geom_col(
position = position_dodge(width = 0.8),
width = 0.7
) +
geom_text(
aes(label = round(Value, 4)),
position = position_dodge(width = 0.8),
vjust = -0.4,
size = 4
) +
labs(
title = "Comparison of ANN Performance Across Wave 3 and Wave 6",
subtitle = "CCivilService in ZAF",
x = "Performance Metric",
y = "Metric Value",
fill = "Wave"
) +
ylim(0, 0.9) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(face = "bold"),
plot.subtitle = element_text(face = "bold"),
axis.title = element_text(face = "bold"),
axis.text.x = element_text(face = "bold"),
legend.title = element_text(face = "bold")
)
# Extract ROC objects for each selected wave
roc_wave_6_q12 = q12_wave_roc_objects[["6"]]
roc_wave_3_q12 = q12_wave_roc_objects[["3"]]
# Extract AUC values for each selected wave
auc_wave_6_q12 = q12_wave_results %>%
filter(Wave == 6) %>%
pull(AUC)
auc_wave_3_q12 = q12_wave_results %>%
filter(Wave == 3) %>%
pull(AUC)
# Plot ROC curve for Wave 6
plot(
roc_wave_6_q12,
col = "blue",
lwd = 2,
main = "ROC Curve Comparison Across Wave 3 and Wave 6",
legacy.axes = TRUE,
xlim = c(1, 0),
ylim = c(0, 1),
xaxs = "i",
yaxs = "i"
)
# Add ROC curve for Wave 3
plot(
roc_wave_3_q12,
col = "red",
lwd = 2,
add = TRUE
)
# Add random-classifier reference line
abline(
a = 0,
b = 1,
lty = 2,
col = "grey"
)
# Add legend with AUC values
legend(
"bottomright",
legend = c(
paste("Wave 6, AUC =", round(auc_wave_6_q12, 4)),
paste("Wave 3, AUC =", round(auc_wave_3_q12, 4))
),
col = c("blue", "red"),
lwd = 2
)
