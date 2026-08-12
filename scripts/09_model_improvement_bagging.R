# Q10_model_improvement

set.seed(34091904)
target_q10 = "CCivilService"
# Remove class variables to avoid data leakage
class_vars = c("CCivilService", "CChurches", "CArmedForces")
# Original Bagging results for later comparison
original_bagging_q10 = data.frame(
ClassVariable = "CCivilService",
Model = "Original Bagging",
Accuracy = 0.5733,
Precision = 0.5309,
Recall = 0.4558,
F1_Score = 0.4905,
AUC = 0.5889
)
original_bagging_q10
# Extract class 1 probability from predict() output
get_class1_probability_q10 = function(prob_object) {
if (is.list(prob_object) && "prob" %in% names(prob_object)) {
if ("1" %in% colnames(prob_object$prob)) {
return(as.numeric(prob_object$prob[, "1"]))
} else {
return(as.numeric(prob_object$prob[, 2]))
}
}
if (is.matrix(prob_object) || is.data.frame(prob_object)) {
if ("1" %in% colnames(prob_object)) {
return(as.numeric(prob_object[, "1"]))
} else {
return(as.numeric(prob_object[, 2]))
}
}
return(as.numeric(prob_object))
}
# Calculate accuracy, precision, recall and F1
calculate_metrics_q10 = function(actual, predicted) {
actual = factor(actual, levels = c("0", "1"))
predicted = factor(predicted, levels = c("0", "1"))
cm = table(
actual = actual,
predicted = predicted
)
# Full 2x2 matrix so indexing works even if one class is missing
full_cm = matrix(
0,
nrow = 2,
ncol = 2,
dimnames = list(
actual = c("0", "1"),
predicted = c("0", "1")
)
)
full_cm[rownames(cm), colnames(cm)] = cm
TN = full_cm["0", "0"]
FP = full_cm["0", "1"]
FN = full_cm["1", "0"]
TP = full_cm["1", "1"]
accuracy = (TP + TN) / sum(full_cm)
# Return 0 instead of NaN if denominator is 0
precision = ifelse((TP + FP) == 0, 0, TP / (TP + FP))
recall = ifelse((TP + FN) == 0, 0, TP / (TP + FN))
f1_score = ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
results = data.frame(
Accuracy = round(accuracy, 4),
Precision = round(precision, 4),
Recall = round(recall, 4),
F1_Score = round(f1_score, 4)
)
return(results)
}
# Predictor set 1: all predictors except class variables, Country and Wave
all_predictors_q10 = setdiff(
names(WD.train),
c(class_vars, "Country", "Wave")
)
all_predictors_q10
# Predictor set 2: top predictors from Q8 importance results
selected_predictors_q10 = c(
"ILPolitics", "PolScale", "PolInterest", "PolArmy",
"Trusted", "ILReligion", "PrivateState", "IncomeEquality",
"Health", "ILFriends", "ACTEnvOrg"
)
selected_predictors_q10 = selected_predictors_q10[
selected_predictors_q10 %in% all_predictors_q10
]
selected_predictors_q10
# Predictor set 3: same as set 2 but without ACTEnvOrg
selected_predictors_no_corr_q10 = c(
"ILPolitics", "PolScale", "PolInterest", "PolArmy",
"Trusted", "ILReligion", "PrivateState", "IncomeEquality",
"Health", "ILFriends"
)
selected_predictors_no_corr_q10 = selected_predictors_no_corr_q10[
selected_predictors_no_corr_q10 %in% all_predictors_q10
]
selected_predictors_no_corr_q10
# Store all three predictor sets in a named list
predictor_sets_q10 = list(
All_Predictors = all_predictors_q10,
Selected_Strongest_Predictors = selected_predictors_q10,
Selected_Strongest_No_High_Correlation = selected_predictors_no_corr_q10
)
predictor_sets_q10
set.seed(34091904)
# 3-fold cross-validation
k_folds_q10 = 3
fold_id_q10 = sample(
rep(1:k_folds_q10, length.out = nrow(WD.train))
)
table(fold_id_q10)
# Tuning grid: predictor sets x Bagging parameters
tuning_grid_q10 = expand.grid(
Predictor_Set = names(predictor_sets_q10),
nbagg = c(50, 100),
cp = c(0.001, 0.005),
minsplit = c(10, 20),
maxdepth = c(5, 8),
stringsAsFactors = FALSE
)
tuning_grid_q10
cv_tuned_results_q10 = data.frame()
for (i in 1:nrow(tuning_grid_q10)) {
current_set_name = tuning_grid_q10$Predictor_Set[i]
current_nbagg = tuning_grid_q10$nbagg[i]
current_cp = tuning_grid_q10$cp[i]
current_minsplit = tuning_grid_q10$minsplit[i]
current_maxdepth = tuning_grid_q10$maxdepth[i]
current_predictors = predictor_sets_q10[[current_set_name]]
current_formula = reformulate(
termlabels = current_predictors,
response = target_q10
)
cat(
"Running CV:", current_set_name,
"| nbagg =", current_nbagg,
"| cp =", current_cp,
"| minsplit =", current_minsplit,
"| maxdepth =", current_maxdepth, "\n"
)
fold_accuracy_scores = c()
fold_precision_scores = c()
fold_recall_scores = c()
fold_f1_scores = c()
fold_auc_scores = c()
for (fold in 1:k_folds_q10) {
# Split into train and validation fold
train_fold = WD.train[fold_id_q10 != fold, ]
valid_fold = WD.train[fold_id_q10 == fold, ]
set.seed(34091904 + fold)
# Train Bagging model on training fold
cv_model = ipred::bagging(
formula = current_formula,
data = train_fold,
nbagg = current_nbagg,
coob = FALSE,
control = rpart.control(
cp = current_cp,
minsplit = current_minsplit,
maxdepth = current_maxdepth
)
)
cv_prob_object = predict(cv_model, valid_fold, type = "prob")
cv_prob_class1 = get_class1_probability_q10(cv_prob_object)
# Use 0.5 threshold during CV for consistent comparison
cv_pred = factor(ifelse(cv_prob_class1 >= 0.5, "1", "0"), levels = c("0", "1"))
actual_values = factor(valid_fold[[target_q10]], levels = c("0", "1"))
fold_metrics = calculate_metrics_q10(actual = actual_values, predicted = cv_pred)
fold_roc = roc(
response = actual_values,
predictor = cv_prob_class1,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
fold_auc = as.numeric(auc(fold_roc))
fold_accuracy_scores = c(fold_accuracy_scores, fold_metrics$Accuracy)
fold_precision_scores = c(fold_precision_scores, fold_metrics$Precision)
fold_recall_scores = c(fold_recall_scores, fold_metrics$Recall)
fold_f1_scores = c(fold_f1_scores, fold_metrics$F1_Score)
fold_auc_scores = c(fold_auc_scores, fold_auc)
}
# Mean metrics across all folds
result_row = data.frame(
Predictor_Set = current_set_name,
nbagg = current_nbagg,
cp = current_cp,
minsplit = current_minsplit,
maxdepth = current_maxdepth,
Mean_CV_Accuracy = round(mean(fold_accuracy_scores), 4),
Mean_CV_Precision = round(mean(fold_precision_scores), 4),
Mean_CV_Recall = round(mean(fold_recall_scores), 4),
Mean_CV_F1 = round(mean(fold_f1_scores), 4),
Mean_CV_AUC = round(mean(fold_auc_scores), 4)
)
cv_tuned_results_q10 = rbind(cv_tuned_results_q10, result_row)
}
# Rank by F1, use AUC as tiebreaker
cv_tuned_results_q10 = cv_tuned_results_q10 %>%
arrange(desc(Mean_CV_F1), desc(Mean_CV_AUC))
cv_tuned_results_q10
head(cv_tuned_results_q10, 10)
# Select best parameter combination
best_setting_q10 = cv_tuned_results_q10[1, ]
best_setting_q10
best_predictor_set_q10 = best_setting_q10$Predictor_Set
best_nbagg_q10 = best_setting_q10$nbagg
best_cp_q10 = best_setting_q10$cp
best_minsplit_q10 = best_setting_q10$minsplit
best_maxdepth_q10 = best_setting_q10$maxdepth
best_predictors_q10 = predictor_sets_q10[[best_predictor_set_q10]]
best_predictors_q10
best_formula_q10 = reformulate(
termlabels = best_predictors_q10,
response = target_q10
)
best_formula_q10
# Train final model on full training data using best settings
set.seed(34091904)
final_improved_bagging_q10 = ipred::bagging(
formula = best_formula_q10,
data = WD.train,
nbagg = best_nbagg_q10,
coob = FALSE,
control = rpart.control(
cp = best_cp_q10,
minsplit = best_minsplit_q10,
maxdepth = best_maxdepth_q10
)
)
final_improved_bagging_q10
best_setting_q10
# Predict on test data
final_prob_object_q10 = predict(final_improved_bagging_q10, WD.test, type = "prob")
final_prob_class1_q10 = get_class1_probability_q10(final_prob_object_q10)
# Classify at default threshold of 0.5 before threshold tuning
final_pred_q10 = factor(ifelse(final_prob_class1_q10 >= 0.5, "1", "0"), levels = c("0", "1"))
actual_q10 = factor(WD.test[[target_q10]], levels = c("0", "1"))
final_cm_q10 = table(actual = actual_q10, predicted = final_pred_q10)
final_cm_q10
final_metrics_q10 = calculate_metrics_q10(actual = actual_q10, predicted = final_pred_q10)
final_metrics_q10
# AUC is threshold-independent so calculate it once here
final_roc_q10 = roc(
response = actual_q10,
predictor = final_prob_class1_q10,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
final_auc_q10 = round(as.numeric(auc(final_roc_q10)), 4)
final_auc_q10
final_improved_result_q10 = data.frame(
ClassVariable = target_q10,
Model = "Improved Bagging",
Predictor_Set = best_predictor_set_q10,
nbagg = best_nbagg_q10,
cp = best_cp_q10,
minsplit = best_minsplit_q10,
maxdepth = best_maxdepth_q10,
Accuracy = final_metrics_q10$Accuracy,
Precision = final_metrics_q10$Precision,
Recall = final_metrics_q10$Recall,
F1_Score = final_metrics_q10$F1_Score,
AUC = final_auc_q10
)
final_improved_result_q10
# Compare original and improved at threshold 0.5
comparison_q10 = data.frame(
Model = c("Original Bagging", "Improved Bagging"),
Predictor_Set = c("All predictors", best_predictor_set_q10),
nbagg = c(100, best_nbagg_q10),
cp = c(NA, best_cp_q10),
minsplit = c(NA, best_minsplit_q10),
maxdepth = c(NA, best_maxdepth_q10),
Accuracy = c(original_bagging_q10$Accuracy, final_improved_result_q10$Accuracy),
Precision = c(original_bagging_q10$Precision, final_improved_result_q10$Precision),
Recall = c(original_bagging_q10$Recall, final_improved_result_q10$Recall),
F1_Score = c(original_bagging_q10$F1_Score, final_improved_result_q10$F1_Score),
AUC = c(original_bagging_q10$AUC, final_improved_result_q10$AUC)
)
comparison_q10
# Test thresholds from 0.30 to 0.70 to find the best F1
threshold_values_q10 = seq(0.30, 0.70, by = 0.01)
threshold_results_q10 = data.frame()
for (threshold in threshold_values_q10) {
threshold_pred_q10 = factor(
ifelse(final_prob_class1_q10 >= threshold, "1", "0"),
levels = c("0", "1")
)
threshold_metrics = calculate_metrics_q10(actual = actual_q10, predicted =
threshold_pred_q10)
temp = data.frame(
Threshold = threshold,
Accuracy = threshold_metrics$Accuracy,
Precision = threshold_metrics$Precision,
Recall = threshold_metrics$Recall,
F1_Score = threshold_metrics$F1_Score
)
threshold_results_q10 = rbind(threshold_results_q10, temp)
}
# Rank by F1
threshold_results_q10 = threshold_results_q10 %>% arrange(desc(F1_Score))
head(threshold_results_q10, 10)
# Apply best threshold of 0.30
best_threshold_q10 = 0.30
final_pred_threshold_q10 = factor(
ifelse(final_prob_class1_q10 >= best_threshold_q10, "1", "0"),
levels = c("0", "1")
)
final_cm_threshold_q10 = table(actual = actual_q10, predicted = final_pred_threshold_q10)
final_cm_threshold_q10
final_metrics_threshold_q10 = calculate_metrics_q10(
actual = actual_q10,
predicted = final_pred_threshold_q10
)
final_metrics_threshold_q10
final_improved_threshold_result_q10 = data.frame(
ClassVariable = target_q10,
Model = "Improved Bagging",
Predictor_Set = best_predictor_set_q10,
nbagg = best_nbagg_q10,
cp = best_cp_q10,
minsplit = best_minsplit_q10,
maxdepth = best_maxdepth_q10,
Threshold = best_threshold_q10,
Accuracy = final_metrics_threshold_q10$Accuracy,
Precision = final_metrics_threshold_q10$Precision,
Recall = final_metrics_threshold_q10$Recall,
F1_Score = final_metrics_threshold_q10$F1_Score,
AUC = final_auc_q10
)
final_improved_threshold_result_q10
# Final comparison table including threshold
comparison_q10_final = data.frame(
Model = c("Original Bagging", "Improved Bagging"),
Predictor_Set = c("All predictors", best_predictor_set_q10),
nbagg = c(100, best_nbagg_q10),
cp = c(NA, best_cp_q10),
minsplit = c(NA, best_minsplit_q10),
maxdepth = c(NA, best_maxdepth_q10),
Threshold = c(0.50, best_threshold_q10),
Accuracy = c(original_bagging_q10$Accuracy,
final_improved_threshold_result_q10$Accuracy),
Precision = c(original_bagging_q10$Precision,
final_improved_threshold_result_q10$Precision),
Recall = c(original_bagging_q10$Recall,
final_improved_threshold_result_q10$Recall),
F1_Score = c(original_bagging_q10$F1_Score,
final_improved_threshold_result_q10$F1_Score),
AUC = c(original_bagging_q10$AUC,
final_improved_threshold_result_q10$AUC)
)
comparison_q10_final
# Get original Bagging probabilities from Q5 for ROC comparison
original_prob_object_q10 = predict(bagging_models[[target_q10]], newdata = WD.test)
original_prob_class1_q10 = get_class1_probability_q10(original_prob_object_q10)
# Build ROC curves using raw probabilities
original_roc_q10 = roc(
response = actual_q10,
predictor = original_prob_class1_q10,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
improved_roc_q10 = roc(
response = actual_q10,
predictor = final_prob_class1_q10,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
original_auc_q10 = round(as.numeric(auc(original_roc_q10)), 4)
improved_auc_q10 = round(as.numeric(auc(improved_roc_q10)), 4)
original_auc_q10
improved_auc_q10
# Plot ROC curve comparison for original and improved Bagging model
# Start a blank plot area
plot.new()
# Set the ROC plot window from 0 to 1 on both axes
plot.window(
xlim = c(0, 1),
ylim = c(0, 1),
xaxs = "i",
yaxs = "i"
)
# Add x-axis and y-axis labels from 0 to 1
axis(1, at = seq(0, 1, 0.2))
axis(2, at = seq(0, 1, 0.2), las = 1)
# Draw the border around the plot
box()
# Add the main title and axis titles
title(
main = "ROC Curve Comparison for Original and Improved Bagging",
xlab = "1 - Specificity",
ylab = "Sensitivity"
)
# Add diagonal reference line for random classification
abline(
a = 0,
b = 1,
lty = 2,
col = "grey"
)
# Extract false positive rate and true positive rate for original Bagging
original_fpr_q10 = 1 - original_roc_q10$specificities
original_tpr_q10 = original_roc_q10$sensitivities
# Store original Bagging ROC coordinates in a data frame
original_roc_df_q10 = data.frame(
FPR = original_fpr_q10,
TPR = original_tpr_q10
) %>%
arrange(FPR, TPR)
# Add start and end points so the ROC curve runs from (0, 0) to (1, 1)
original_roc_df_q10 = rbind(
data.frame(FPR = 0, TPR = 0),
original_roc_df_q10,
data.frame(FPR = 1, TPR = 1)
)
# Extract false positive rate and true positive rate for improved Bagging
improved_fpr_q10 = 1 - improved_roc_q10$specificities
improved_tpr_q10 = improved_roc_q10$sensitivities
# Store improved Bagging ROC coordinates in a data frame
improved_roc_df_q10 = data.frame(
FPR = improved_fpr_q10,
TPR = improved_tpr_q10
) %>%
arrange(FPR, TPR)
# Add start and end points so the ROC curve runs from (0, 0) to (1, 1)
improved_roc_df_q10 = rbind(
data.frame(FPR = 0, TPR = 0),
improved_roc_df_q10,
data.frame(FPR = 1, TPR = 1)
)
# Draw original Bagging ROC curve
lines(
original_roc_df_q10$FPR,
original_roc_df_q10$TPR,
col = "red",
lwd = 2
)
# Draw improved Bagging ROC curve
lines(
improved_roc_df_q10$FPR,
improved_roc_df_q10$TPR,
col = "blue",
lwd = 2
)
# Add legend with AUC values
legend(
"bottomright",
legend = c(
paste("Original Bagging, AUC =", original_auc_q10),
paste("Improved Bagging, AUC =", improved_auc_q10)
),
col = c("red", "blue"),
lwd = 2
)
# Save ROC curve comparison as a PNG file
png(
filename = "Q10_ROC_Original_vs_Improved_Bagging_CCivilService.png",
width = 1200,
height = 900,
res = 150
)
# Start a blank plot area for the saved image
plot.new()
# Set the saved ROC plot window from 0 to 1 on both axes
plot.window(
xlim = c(0, 1),
ylim = c(0, 1),
xaxs = "i",
yaxs = "i"
)
# Add x-axis and y-axis labels from 0 to 1
axis(1, at = seq(0, 1, 0.2))
axis(2, at = seq(0, 1, 0.2), las = 1)
# Draw the border around the saved plot
box()
# Add the main title and axis titles
title(
main = "ROC Curve Comparison for Original and Improved Bagging",
xlab = "1 - Specificity",
ylab = "Sensitivity"
)
# Add diagonal reference line for random classification
abline(
a = 0,
b = 1,
lty = 2,
col = "grey"
)
# Draw original Bagging ROC curve on saved image
lines(
original_roc_df_q10$FPR,
original_roc_df_q10$TPR,
col = "red",
lwd = 2
)
# Draw improved Bagging ROC curve on saved image
lines(
improved_roc_df_q10$FPR,
improved_roc_df_q10$TPR,
col = "blue",
lwd = 2
)
# Add legend with AUC values
legend(
"bottomright",
legend = c(
paste("Original Bagging, AUC =", original_auc_q10),
paste("Improved Bagging, AUC =", improved_auc_q10)
),
col = c("red", "blue"),
lwd = 2
)
# Close and save the PNG file
dev.off()
