# Q11_ann_classifier

set.seed(34091904)
# Count observations per country to identify ZAF as the largest
country_counts_q11 = WD_clean %>%
group_by(Country) %>%
summarise(Number_Observations = n(), .groups = "drop") %>%
arrange(desc(Number_Observations))
country_counts_q11
largest_country_q11 = country_counts_q11$Country[1]
largest_country_q11
# Highlight ZAF in red, all others in blue
country_counts_q11 = country_counts_q11 %>%
arrange(desc(Number_Observations)) %>%
mutate(
Highlight = ifelse(Country == "ZAF", "ZAF", "Other"),
Country = factor(Country, levels = rev(Country))
)
ggplot(country_counts_q11, aes(x = Country, y = Number_Observations, fill = Highlight)) +
geom_col() +
geom_text(aes(label = Number_Observations), hjust = -0.1, size = 3) +
coord_flip() +
scale_fill_manual(values = c("ZAF" = "firebrick3", "Other" = "steelblue"), guide = "none") +
labs(
title = "Number of Observations by Country",
subtitle = "ZAF has the highest number of observations and was selected for Question
11",
x = "Country",
y = "Number of Observations"
) +
theme_minimal() +
theme(
plot.title = element_text(face = "bold", size = 16),
plot.subtitle = element_text(size = 12),
axis.title.x = element_text(face = "bold", size = 12),
axis.title.y = element_text(face = "bold", size = 12),
axis.text.x = element_text(face = "bold", size = 10),
axis.text.y = element_text(face = "bold", size = 9)
)
# Target variable and selected country
target_q11 = "CCivilService"
country_q11 = "ZAF"
# Filter existing train/test split to ZAF only
# Do not re-split - use the same split from Q3
train_country_q11 = WD.train %>% filter(Country == country_q11)
test_country_q11 = WD.test %>% filter(Country == country_q11)
nrow(train_country_q11)
nrow(test_country_q11)
table(train_country_q11[[target_q11]])
table(test_country_q11[[target_q11]])
# Class distribution for ZAF training and test sets
zaf_distribution_q11 = data.frame(
Dataset = c("Training", "Training", "Test", "Test"),
Class = c("Low confidence (0)", "High confidence (1)",
"Low confidence (0)", "High confidence (1)"),
Count = c(
as.numeric(table(train_country_q11[[target_q11]])["0"]),
as.numeric(table(train_country_q11[[target_q11]])["1"]),
as.numeric(table(test_country_q11[[target_q11]])["0"]),
as.numeric(table(test_country_q11[[target_q11]])["1"])
)
)
zaf_distribution_q11
# Selected predictors based on CCivilService importance results from Q8
selected_predictors_q11 = c(
"PolScale", "PrivateState", "IncomeEquality", "ILPolitics",
"PolInterest", "PolArmy", "ILReligion", "PolDemoc",
"Health", "Trusted", "ILFriends", "ILLeisure",
"PolPetition", "FutureWork"
)
selected_predictors_q11 = selected_predictors_q11[
selected_predictors_q11 %in% names(train_country_q11)
]
# All valid predictors: remove class variables, Country and Wave to avoid leakage
class_vars_q11 = grep("^C", names(train_country_q11), value = TRUE)
excluded_vars_q11 = unique(c(target_q11, class_vars_q11, "Country", "Wave"))
all_predictors_q11 = setdiff(names(train_country_q11), excluded_vars_q11)
all_predictors_q11 = all_predictors_q11[all_predictors_q11 %in% names(train_country_q11)]
selected_predictors_q11
all_predictors_q11
# Calculate accuracy, precision, recall and F1
# Builds full 2x2 matrix to handle missing classes safely
calculate_ann_metrics_q11 = function(actual, predicted) {
# Convert to factors with fixed levels
actual = factor(actual, levels = c("0", "1"))
predicted = factor(predicted, levels = c("0", "1"))
# Build confusion matrix
cm = table(Actual = actual, Predicted = predicted)
# Add missing rows/columns so the matrix is always 2x2
if (!("0" %in% rownames(cm))) cm = rbind(cm, "0" = c("0" = 0, "1" = 0))
if (!("1" %in% rownames(cm))) cm = rbind(cm, "1" = c("0" = 0, "1" = 0))
if (!("0" %in% colnames(cm))) cm = cbind(cm, "0" = c("0" = 0, "1" = 0))
if (!("1" %in% colnames(cm))) cm = cbind(cm, "1" = c("0" = 0, "1" = 0))
# Reorder to ensure correct TN/FP/FN/TP positions
cm = cm[c("0", "1"), c("0", "1")]
# Extract confusion matrix components
tn = cm["0", "0"]
fp = cm["0", "1"]
fn = cm["1", "0"]
tp = cm["1", "1"]
# Calculate metrics
accuracy = (tp + tn) / sum(cm)
# Return 0 instead of NaN if denominator is 0
precision = ifelse((tp + fp) == 0, 0, tp / (tp + fp))
recall = ifelse((tp + fn) == 0, 0, tp / (tp + fn))
f1_score = ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
data.frame(
Accuracy = round(as.numeric(accuracy), 4),
Precision = round(as.numeric(precision), 4),
Recall = round(as.numeric(recall), 4),
F1_Score = round(as.numeric(f1_score), 4)
)
}
# Prepare ANN data: dummy code + scale using training statistics only
# Test data is scaled using training mean and SD to prevent leakage
prepare_ann_data_q11 = function(train_data, test_data, predictors, target) {
# Keep only predictor columns and the target
needed_cols = c(predictors, target)
# Drop rows with missing values
train_data = train_data %>% select(all_of(needed_cols)) %>% na.omit()
test_data = test_data %>% select(all_of(needed_cols)) %>% na.omit()
# Extract target labels as factors
y_train = factor(as.character(train_data[[target]]), levels = c("0", "1"))
y_test = factor(as.character(test_data[[target]]), levels = c("0", "1"))
# Remove target column from predictor data frames
train_predictors = train_data; train_predictors[[target]] = NULL
test_predictors = test_data; test_predictors[[target]] = NULL
# Combine train and test predictors so dummy coding is consistent across both
combined_predictors = rbind(train_predictors, test_predictors)
# Build dummy coding formula from predictor names
dummy_formula = reformulate(termlabels = predictors, response = NULL)
# Apply dummy coding to combined data
x_combined = model.matrix(dummy_formula, data = combined_predictors)
# Remove intercept column added by model.matrix
x_combined = x_combined[, colnames(x_combined) != "(Intercept)", drop = FALSE]
# Split back into train and test
x_train = x_combined[1:nrow(train_predictors), , drop = FALSE]
x_test = x_combined[(nrow(train_predictors) + 1):nrow(x_combined), , drop = FALSE]
# Compute scaling parameters from training data only
train_means = apply(x_train, 2, mean)
train_sds = apply(x_train, 2, sd)
# Replace zero SDs with 1 to avoid division by zero on constant columns
train_sds[train_sds == 0] = 1
# Scale both sets using training statistics
x_train_scaled = scale(x_train, center = train_means, scale = train_sds)
x_test_scaled = scale(x_test, center = train_means, scale = train_sds)
list(
x_train = x_train_scaled,
x_test = x_test_scaled,
y_train = y_train,
y_test = y_test,
train_rows = train_data,
test_rows = test_data
)
}
# Find the classification threshold that maximises F1-score
find_best_threshold_q11 = function(actual, probabilities) {
# Test thresholds from 0.10 to 0.90 in steps of 0.01
thresholds = seq(0.10, 0.90, by = 0.01)
# Empty results table
threshold_results = data.frame(
Threshold = thresholds,
Accuracy = NA, Precision = NA, Recall = NA, F1_Score = NA
)
for (i in seq_along(thresholds)) {
# Classify using current threshold
predicted = factor(
ifelse(probabilities >= thresholds[i], "1", "0"),
levels = c("0", "1")
)
# Calculate metrics at this threshold
current_metrics = calculate_ann_metrics_q11(actual = actual, predicted = predicted)
threshold_results$Accuracy[i] = current_metrics$Accuracy
threshold_results$Precision[i] = current_metrics$Precision
threshold_results$Recall[i] = current_metrics$Recall
threshold_results$F1_Score[i] = current_metrics$F1_Score
}
# Return the threshold with the best F1, using recall then precision as tiebreakers
threshold_results %>%
arrange(desc(F1_Score), desc(Recall), desc(Precision)) %>%
slice(1)
}
# Full ANN tuning pipeline for one predictor set
# Uses 80:20 internal validation split, tunes size/decay/maxit, then refits on full training data
tune_ann_model_q11 = function(train_data, test_data, predictors, target, model_name) {
# Prepare dummy-coded and scaled data
ann_data = prepare_ann_data_q11(train_data, test_data, predictors, target)
x_train = ann_data$x_train; x_test = ann_data$x_test
y_train = ann_data$y_train; y_test = ann_data$y_test
set.seed(34091904)
# 80:20 internal validation split for hyperparameter tuning
train_index = sample(1:nrow(x_train), size = floor(0.80 * nrow(x_train)))
x_fit = x_train[ train_index, , drop = FALSE]; y_fit = y_train[ train_index]
x_valid = x_train[-train_index, , drop = FALSE]; y_valid = y_train[-train_index]
# Convert labels to indicator matrix for nnet
y_fit_ann = class.ind(y_fit)
# Class weights to handle imbalance
class_counts = table(y_fit)
weights_fit = ifelse(
y_fit == "1",
sum(class_counts) / (2 * class_counts["1"]),
sum(class_counts) / (2 * class_counts["0"])
)
# Tuning grid: hidden layer size, weight decay, max iterations
tuning_grid = expand.grid(
size = c(3, 5, 7, 10, 12),
decay = c(0, 0.001, 0.01, 0.05, 0.10),
maxit = c(500, 1000),
stringsAsFactors = FALSE
)
tuning_results = data.frame()
best_f1 = -1; best_threshold = 0.5; best_settings = NULL
for (i in 1:nrow(tuning_grid)) {
current_size = tuning_grid$size[i]
current_decay = tuning_grid$decay[i]
current_maxit = tuning_grid$maxit[i]
# Use a different reproducible seed for each tuning setting
# so each ANN starts with a controlled but not identical random initialisation
set.seed(34091904 + i)
# Fit ANN on training fold, skip if error
current_model = tryCatch({
nnet(
x = x_fit, y = y_fit_ann,
size = current_size,
decay = current_decay,
maxit = current_maxit,
softmax = TRUE,
trace = FALSE,
weights = weights_fit
)
}, error = function(e) NULL)
if (!is.null(current_model)) {
# Get predicted probabilities on validation set
valid_prob = predict(current_model, x_valid, type = "raw")
valid_prob_class1 = valid_prob[, "1"]
# Find best threshold on validation set
best_threshold_result = find_best_threshold_q11(
actual = y_valid,
probabilities = valid_prob_class1
)
# Classify using best threshold
valid_pred = factor(
ifelse(valid_prob_class1 >= best_threshold_result$Threshold, "1", "0"),
levels = c("0", "1")
)
# Calculate validation metrics
valid_metrics = calculate_ann_metrics_q11(actual = y_valid, predicted = valid_pred)
# Calculate validation AUC
valid_auc = tryCatch({
round(as.numeric(auc(roc(
response = y_valid,
predictor = valid_prob_class1,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
))), 4)
}, error = function(e) NA)
# Store results for this combination
result_row = data.frame(
Model_Set = model_name,
Size = current_size,
Decay = current_decay,
Maxit = current_maxit,
Threshold = best_threshold_result$Threshold,
Validation_Accuracy = valid_metrics$Accuracy,
Validation_Precision = valid_metrics$Precision,
Validation_Recall = valid_metrics$Recall,
Validation_F1 = valid_metrics$F1_Score,
Validation_AUC = valid_auc
)
tuning_results = rbind(tuning_results, result_row)
# Track best F1 seen so far
if (valid_metrics$F1_Score > best_f1) {
best_f1 = valid_metrics$F1_Score
best_threshold = best_threshold_result$Threshold
best_settings = result_row
}
}
}
# Rank all combinations by F1 then AUC
tuning_results = tuning_results %>%
arrange(desc(Validation_F1), desc(Validation_AUC), desc(Validation_Accuracy))
best_settings = tuning_results[1, ]
# Refit final model on full training data using best settings
y_train_ann = class.ind(y_train)
full_class_counts = table(y_train)
# Recalculate class weights using full training set
full_weights = ifelse(
y_train == "1",
sum(full_class_counts) / (2 * full_class_counts["1"]),
sum(full_class_counts) / (2 * full_class_counts["0"])
)
set.seed(34091904)
final_model = nnet(
x = x_train, y = y_train_ann,
size = best_settings$Size,
decay = best_settings$Decay,
maxit = best_settings$Maxit,
softmax = TRUE,
trace = FALSE,
weights = full_weights
)
# Predict on test set using best threshold
test_prob = predict(final_model, x_test, type = "raw")
test_prob_class1 = test_prob[, "1"]
test_pred = factor(
ifelse(test_prob_class1 >= best_settings$Threshold, "1", "0"),
levels = c("0", "1")
)
# Calculate test metrics
test_metrics = calculate_ann_metrics_q11(actual = y_test, predicted = test_pred)
# Calculate test AUC
test_auc = tryCatch({
round(as.numeric(auc(roc(
response = y_test,
predictor = test_prob_class1,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
))), 4)
}, error = function(e) NA)
# Store final result
final_result = data.frame(
Country = country_q11,
ClassVariable = target_q11,
Model_Set = model_name,
Number_Predictors = length(predictors),
Number_Dummy_Columns = ncol(x_train),
Best_Size = best_settings$Size,
Best_Decay = best_settings$Decay,
Best_Maxit = best_settings$Maxit,
Best_Threshold = best_settings$Threshold,
Accuracy = test_metrics$Accuracy,
Precision = test_metrics$Precision,
Recall = test_metrics$Recall,
F1_Score = test_metrics$F1_Score,
AUC = test_auc
)
list(
final_model = final_model,
final_result = final_result,
tuning_results = tuning_results,
confusion_matrix = table(Actual = y_test, Predicted = test_pred),
probabilities = test_prob_class1,
predictions = test_pred,
actual = y_test,
x_test = x_test,
x_train = x_train,
y_test = y_test,
y_train = y_train,
test_rows = ann_data$test_rows,
predictors = predictors
)
}
# Run ANN with selected predictors
ann_selected_q11 = tune_ann_model_q11(
train_data = train_country_q11, test_data = test_country_q11,
predictors = selected_predictors_q11, target = target_q11,
model_name = "ANN - Selected Predictors"
)
ann_selected_q11$final_result
ann_selected_q11$confusion_matrix
head(ann_selected_q11$tuning_results)
# Run ANN with all predictors
ann_all_q11 = tune_ann_model_q11(
train_data = train_country_q11, test_data = test_country_q11,
predictors = all_predictors_q11, target = target_q11,
model_name = "ANN - All Predictors"
)
ann_all_q11$final_result
ann_all_q11$confusion_matrix
head(ann_all_q11$tuning_results)
# Compare both predictor sets - rank by F1 then AUC
ann_comparison_q11 = rbind(ann_selected_q11$final_result, ann_all_q11$final_result) %>%
arrange(desc(F1_Score), desc(AUC), desc(Accuracy))
ann_comparison_q11
# Select best setting from the all-predictor ANN tuning results
best_ann_cv_setting_q11 = ann_all_q11$tuning_results[1, ]
best_ann_cv_setting_q11
# Extract best parameters
best_size_q11 = best_ann_cv_setting_q11$Size
best_decay_q11 = best_ann_cv_setting_q11$Decay
best_maxit_q11 = best_ann_cv_setting_q11$Maxit
best_threshold_q11 = best_ann_cv_setting_q11$Threshold
# Convert labels to indicator matrix for nnet
y_train_ann_all_q11 = class.ind(ann_all_q11$y_train)
# Class weights using full ZAF training set
full_class_counts_q11 = table(ann_all_q11$y_train)
final_weights_q11 = ifelse(
ann_all_q11$y_train == "1",
sum(full_class_counts_q11) / (2 * full_class_counts_q11["1"]),
sum(full_class_counts_q11) / (2 * full_class_counts_q11["0"])
)
# Train final ANN on full ZAF training data using best settings
set.seed(34091904)
final_ann_improved_q11 = nnet(
x = ann_all_q11$x_train, y = y_train_ann_all_q11,
size = best_size_q11,
decay = best_decay_q11,
maxit = best_maxit_q11,
softmax = TRUE,
trace = FALSE,
weights = final_weights_q11
)
# Predict on ZAF test data
final_ann_prob_q11 = predict(final_ann_improved_q11, ann_all_q11$x_test, type =
"raw")
final_ann_prob_class1_q11 = final_ann_prob_q11[, "1"]
# Classify using best threshold
final_ann_pred_q11 = factor(
ifelse(final_ann_prob_class1_q11 >= best_threshold_q11, "1", "0"),
levels = c("0", "1")
)
final_ann_cm_q11 = table(Actual = ann_all_q11$y_test, Predicted = final_ann_pred_q11)
final_ann_cm_q11
final_ann_metrics_q11 = calculate_ann_metrics_q11(
actual = ann_all_q11$y_test,
predicted = final_ann_pred_q11
)
# AUC is threshold-independent so calculate from raw probabilities
final_ann_auc_q11 = round(
as.numeric(auc(roc(
response = ann_all_q11$y_test,
predictor = final_ann_prob_class1_q11,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
))), 4
)
final_ann_auc_q11
# Store final result
final_ann_improved_result_q11 = data.frame(
Country = country_q11,
ClassVariable = target_q11,
Model = "Improved ANN - All Predictors",
Number_Predictors = length(all_predictors_q11),
Number_Dummy_Columns = ncol(ann_all_q11$x_train),
Size = best_size_q11,
Decay = best_decay_q11,
Maxit = best_maxit_q11,
Threshold = best_threshold_q11,
Accuracy = final_ann_metrics_q11$Accuracy,
Precision = final_ann_metrics_q11$Precision,
Recall = final_ann_metrics_q11$Recall,
F1_Score = final_ann_metrics_q11$F1_Score,
AUC = final_ann_auc_q11
)
final_ann_improved_result_q11
# Use probabilities from the all-predictor ANN for the ROC curve
ann_actual_q11 = ann_all_q11$actual
ann_prob_q11 = ann_all_q11$probabilities
ann_roc_q11 = roc(
response = ann_actual_q11,
predictor = ann_prob_q11,
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
ann_auc_q11 = round(as.numeric(auc(ann_roc_q11)), 4)
ann_auc_q11
# Plot ROC curve: blue = ANN, grey dashed = random classifier
plot(
ann_roc_q11,
col = "blue", lwd = 2,
main = "ROC Curve for ANN Model Predicting CCivilService",
legacy.axes = TRUE, identity = FALSE
)
abline(a = 0, b = 1, lty = 2, col = "grey")
legend("bottomright", legend = paste("ANN, AUC =", ann_auc_q11), col = "blue", lwd = 2)
# Performance comparison: selected vs all predictors
ann_metric_comparison_q11 = data.frame(
Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "AUC"),
Selected_Predictors = c(0.4915, 0.4318, 0.7917, 0.5588, 0.4940),
All_Predictors = c(0.5763, 0.4865, 0.7500, 0.5902, 0.6476)
)
names(ann_metric_comparison_q11)
# Reshape to long format for ggplot
ann_metric_comparison_long_q11 = ann_metric_comparison_q11 %>%
pivot_longer(
cols = c(Selected_Predictors, All_Predictors),
names_to = "Predictor_Set",
values_to = "Value"
) %>%
mutate(
Predictor_Set = recode(
Predictor_Set,
"Selected_Predictors" = "Selected Predictors",
"All_Predictors" = "All Predictors"
)
)
# Grouped bar chart comparing both predictor sets
ggplot(ann_metric_comparison_long_q11, aes(x = Metric, y = Value, fill = Predictor_Set)) +
geom_col(position = position_dodge(width = 0.8), width = 0.7) +
geom_text(
aes(label = round(Value, 4)),
position = position_dodge(width = 0.8),
vjust = -0.3, size = 4, fontface = "bold"
) +
labs(
title = "Comparison of ANN Performance: Selected vs All Predictors",
x = "Performance Metric",
y = "Metric Value",
fill = "Predictor Set"
) +
ylim(0, 0.9) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
axis.title.x = element_text(size = 14, face = "bold"),
axis.title.y = element_text(size = 14, face = "bold"),
axis.text.x = element_text(size = 12, face = "bold"),
axis.text.y = element_text(size = 12, face = "bold"),
legend.title = element_text(size = 13, face = "bold"),
legend.text = element_text(size = 12)
)
