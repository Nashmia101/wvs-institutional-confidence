# Q5_performance_metrics

Question 5 Classification and Performance Metrics code (
Predictions on test data and confusion matrix was created together
with question 4 code
# Function to calculate accuracy, precision, recall, and F1-score
# Positive class is defined as 1 (High confidence)
calculate_metrics = function(conf_matrix) {
# Initialise a full 2x2 matrix with zeros to handle cases where a class may be missing
full_matrix = matrix(
0,
nrow = 2,
ncol = 2,
dimnames = list(
actual = c("0", "1"),
predicted = c("0", "1")
)
)
# Fill in the available confusion matrix values
full_matrix[rownames(conf_matrix), colnames(conf_matrix)] = conf_matrix
# Extract true negatives, false positives, false negatives, and true positives
TN = full_matrix["0", "0"]
FP = full_matrix["0", "1"]
FN = full_matrix["1", "0"]
TP = full_matrix["1", "1"]
# Calculate accuracy
accuracy = (TP + TN) / sum(full_matrix)
# Calculate precision, returning 0 if no positive predictions were made
precision = ifelse((TP + FP) == 0, 0, TP / (TP + FP))
# Calculate recall, returning 0 if no actual positives exist
recall = ifelse((TP + FN) == 0, 0, TP / (TP + FN))
# Calculate F1-score, returning 0 if both precision and recall are zero
f1_score = ifelse(
(precision + recall) == 0,
0,
2 * precision * recall / (precision + recall)
)
return(c(
Accuracy = round(accuracy, 4),
Precision = round(precision, 4),
Recall = round(recall, 4),
F1_Score = round(f1_score, 4)
))
}
# Combine all model confusion matrices into a single named list for easy iteration
all_confusion_matrices = list(
"Decision Tree" = tree_confusion_matrices,
"Naive Bayes" = nb_confusion_matrices,
"Bagging" = bagging_confusion_matrices,
"Boosting" = boosting_confusion_matrices,
"Random Forest" = rf_confusion_matrices
)
# Loop through each class variable and compute performance metrics for all models
performance_tables = list()
for (target in class_vars) {
results = data.frame()
# Loop through each model and calculate metrics from its confusion matrix
for (model_name in names(all_confusion_matrices)) {
cm = all_confusion_matrices[[model_name]][[target]]
metrics = calculate_metrics(cm)
# Store metrics in a temporary dataframe row
temp = data.frame(
Model = model_name,
Accuracy = metrics["Accuracy"],
Precision = metrics["Precision"],
Recall = metrics["Recall"],
F1_Score = metrics["F1_Score"]
)
results = rbind(results, temp)
}
# Store the completed performance table for this class variable
performance_tables[[target]] = results
}
# Print performance tables for each class variable
performance_tables$CCivilService
performance_tables$CChurches
performance_tables$CArmedForces
# Redefine combined confusion matrix list for confusion table creation
tree_confusion_matrices_q5 = tree_confusion_matrices
tree_confusion_matrices_q5$CArmedForces = cm_CArmedForces_improved
all_confusion_matrices = list(
"Decision Tree" = tree_confusion_matrices_q5 ,
"Naive Bayes" = nb_confusion_matrices,
"Bagging" = bagging_confusion_matrices,
"Boosting" = boosting_confusion_matrices,
"Random Forest" = rf_confusion_matrices
)
# Function to convert a confusion matrix into a flat dataframe row for reporting
confusion_to_df = function(conf_matrix, model_name) {
# Initialise full 2x2 matrix with zeros to handle missing class predictions
full_matrix = matrix(
0,
nrow = 2,
ncol = 2,
dimnames = list(
actual = c("0", "1"),
predicted = c("0", "1")
)
)
# Fill in available values from the confusion matrix
full_matrix[rownames(conf_matrix), colnames(conf_matrix)] = conf_matrix
# Return as a dataframe with labelled columns
data.frame(
Model = model_name,
Actual_0_Predicted_0 = full_matrix["0", "0"],
Actual_0_Predicted_1 = full_matrix["0", "1"],
Actual_1_Predicted_0 = full_matrix["1", "0"],
Actual_1_Predicted_1 = full_matrix["1", "1"]
)
}
# Loop through each class variable and compile confusion matrix summary tables
confusion_tables = list()
for (target in class_vars) {
results = data.frame()
# Loop through each model and convert its confusion matrix to a dataframe row
for (model_name in names(all_confusion_matrices)) {
cm = all_confusion_matrices[[model_name]][[target]]
temp = confusion_to_df(cm, model_name)
results = rbind(results, temp)
}
# Store the completed confusion table for this class variable
confusion_tables[[target]] = results
}
# Print confusion tables for all class variables
Confusion_tables
