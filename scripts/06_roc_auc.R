# Q6_roc_auc

# Helper function to extract class 1 probability from different model output formats
get_class1_probability = function(prob_object) {
# If probabilities are stored in a matrix or data frame, extract class 1 column
if (is.matrix(prob_object) || is.data.frame(prob_object)) {
if ("1" %in% colnames(prob_object)) {
return(prob_object[, "1"])
} else {
return(prob_object[, 2])
}
}
# If probability object is already a vector, return directly
return(prob_object)
}
# Extract prediction probabilities for class 1 from all five models for each class variable
probabilities = list()
for (target in class_vars) {
probabilities[[target]] = list()
# Decision Tree probabilities
# Use improved rpart tree for CArmedForces as default tree produced a single-node model
if (target == "CArmedForces") {
tree_prob = predict(
tree_CArmedForces_improved,
newdata = WD.test,
type = "prob"
)
} else {
# Use default tree models for CCivilService and CChurches
tree_prob = predict(
tree_models[[target]],
WD.test,
type = "vector"
)
}
probabilities[[target]][["Decision Tree"]] = get_class1_probability(tree_prob)
# Naive Bayes probabilities
nb_prob = predict(
nb_models[[target]],
WD.test,
type = "raw"
)
probabilities[[target]][["Naive Bayes"]] = get_class1_probability(nb_prob)
# Bagging probabilities
bagging_prob_object = predict(
bagging_models[[target]],
newdata = WD.test
)
probabilities[[target]][["Bagging"]] = get_class1_probability(
bagging_prob_object$prob
)
# Boosting probabilities
boosting_prob_object = predict(
boosting_models[[target]],
newdata = WD.test
)
probabilities[[target]][["Boosting"]] = get_class1_probability(
boosting_prob_object$prob
)
# Random Forest probabilities
rf_prob = predict(
rf_models[[target]],
WD.test,
type = "prob"
)
probabilities[[target]][["Random Forest"]] = get_class1_probability(rf_prob)
}
# Compute ROC objects and AUC values for all models across each class variable
roc_objects = list()
auc_tables = list()
# Loop through each class variable and each model to compute ROC and AUC
for (target in class_vars) {
roc_objects[[target]] = list()
auc_results = data.frame()
actual_class = WD.test[[target]]
for (model_name in names(probabilities[[target]])) {
# Compute ROC curve with class 0 as negative and class 1 as positive
roc_obj = roc(
response = actual_class,
predictor = probabilities[[target]][[model_name]],
levels = c("0", "1"),
direction = "<",
quiet = TRUE
)
roc_objects[[target]][[model_name]] = roc_obj
# Store AUC value for this model and class variable
temp = data.frame(
ClassVariable = target,
Model = model_name,
AUC = round(as.numeric(auc(roc_obj)), 4)
)
auc_results = rbind(auc_results, temp)
}
auc_tables[[target]] = auc_results
}
# Print AUC tables for each class variable
auc_tables$CCivilService
auc_tables$CChurches
auc_tables$CArmedForces
# Create a function to plot ROC curves for a selected class variable
plot_roc_curves = function(target) {
# Get the names of all models available for the selected class variable
model_names = names(roc_objects[[target]])
# Create an empty ROC plot with fixed axes from 0 to 1
plot(
NA,
xlim = c(0, 1),
ylim = c(0, 1),
xaxs = "i",
yaxs = "i",
xlab = "1 - Specificity",
ylab = "Sensitivity",
main = paste("ROC Curves for", target)
)
# Add diagonal reference line representing random classification
abline(
a = 0,
b = 1,
lty = 2,
col = "gray"
)
# Loop through each model and add its ROC curve to the plot
for (i in 1:length(model_names)) {
# Extract ROC object for the current model
current_roc = roc_objects[[target]][[model_names[i]]]
# Calculate false positive rate and sensitivity values
false_positive_rate = 1 - current_roc$specificities
sensitivity = current_roc$sensitivities
# Draw the ROC curve for the current model
lines(
false_positive_rate,
sensitivity,
col = i,
lwd = 2
)
}
# Add legend showing model names and matching curve colours
legend(
"bottomright",
legend = model_names,
col = 1:length(model_names),
lwd = 2,
cex = 0.8
)
}
# Save ROC curve plot for CCivilService
png("Q6_ROC_CCivilService.png", width = 2400, height = 1800, res = 300)
plot_roc_curves("CCivilService")
dev.off()
# Save ROC curve plot for CChurches
png("Q6_ROC_CChurches.png", width = 2400, height = 1800, res = 300)
plot_roc_curves("CChurches")
dev.off()
# Save ROC curve plot for CArmedForces
png("Q6_ROC_CArmedForces.png", width = 2400, height = 1800, res = 300)
plot_roc_curves("CArmedForces")
dev.off()
# Create a function that converts a confusion matrix into long format
# This makes the confusion matrix suitable for ggplot heatmap plotting
make_confusion_heatmap_data = function(target_class) {
# Create an empty data frame to store all model confusion matrices
heatmap_data = data.frame()
# Loop through each model stored in all_confusion_matrices
for (model_name in names(all_confusion_matrices)) {
# Extract the confusion matrix for the selected class variable
cm = all_confusion_matrices[[model_name]][[target_class]]
# Create a full 2x2 confusion matrix with both classes 0 and 1
# This prevents errors if any model predicts only one class
full_cm = matrix(
0,
nrow = 2,
ncol = 2,
dimnames = list(
Actual = c("0", "1"),
Predicted = c("0", "1")
)
)
# Fill the full matrix with the actual confusion matrix values
full_cm[rownames(cm), colnames(cm)] = cm
# Convert the matrix into long format for ggplot
temp = as.data.frame(as.table(full_cm))
# Add model name and class variable name for faceting and labelling
temp$Model = model_name
temp$ClassVariable = target_class
# Add this model's confusion matrix data to the full heatmap dataset
heatmap_data = rbind(heatmap_data, temp)
}
# Return the completed heatmap dataset
return(heatmap_data)
}
# Create heatmap data for each class variable
heatmap_CCivilService = make_confusion_heatmap_data("CCivilService")
heatmap_CChurches = make_confusion_heatmap_data("CChurches")
heatmap_CArmedForces = make_confusion_heatmap_data("CArmedForces")
# Plot confusion matrix heatmaps for CCivilService across all models
ggplot(
heatmap_CCivilService,
aes(x = Predicted, y = Actual, fill = Freq)
) +
geom_tile(colour = "white", linewidth = 1) +
geom_text(
aes(label = Freq),
size = 5,
fontface = "bold"
) +
facet_wrap(~ Model, nrow = 1) +
labs(
title = "Confusion Matrix Heatmaps for CCivilService",
x = "Predicted Class",
y = "Actual Class",
fill = "Count"
) +
scale_y_discrete(limits = rev) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
strip.text = element_text(face = "bold", size = 12),
axis.title = element_text(face = "bold"),
axis.text = element_text(face = "bold"),
legend.title = element_text(face = "bold")
)
# Plot confusion matrix heatmaps for CChurches across all models
ggplot(
heatmap_CChurches,
aes(x = Predicted, y = Actual, fill = Freq)
) +
geom_tile(colour = "white", linewidth = 1) +
geom_text(
aes(label = Freq),
size = 5,
fontface = "bold"
) +
facet_wrap(~ Model, nrow = 1) +
labs(
title = "Confusion Matrix Heatmaps for CChurches",
x = "Predicted Class",
y = "Actual Class",
fill = "Count"
) +
scale_y_discrete(limits = rev) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
strip.text = element_text(face = "bold", size = 12),
axis.title = element_text(face = "bold"),
axis.text = element_text(face = "bold"),
legend.title = element_text(face = "bold")
)
# Create a separate function for CArmedForces
# This includes both the default and improved Decision Tree models for comparison
make_carmedforces_heatmap_data = function() {
# Create an empty data frame to store CArmedForces confusion matrices
heatmap_data = data.frame()
# Store all CArmedForces confusion matrices in one named list
carmedforces_matrices = list(
"Decision Tree - Default" = tree_confusion_matrices$CArmedForces,
"Decision Tree - Improved" = cm_CArmedForces_improved,
"Naive Bayes" = nb_confusion_matrices$CArmedForces,
"Bagging" = bagging_confusion_matrices$CArmedForces,
"Boosting" = boosting_confusion_matrices$CArmedForces,
"Random Forest" = rf_confusion_matrices$CArmedForces
)
# Loop through each CArmedForces model confusion matrix
for (model_name in names(carmedforces_matrices)) {
# Extract the confusion matrix for the current model
cm = carmedforces_matrices[[model_name]]
# Create a full 2x2 matrix to ensure both classes are shown
full_cm = matrix(
0,
nrow = 2,
ncol = 2,
dimnames = list(
Actual = c("0", "1"),
Predicted = c("0", "1")
)
)
# Add the actual confusion matrix values into the full matrix
full_cm[rownames(cm), colnames(cm)] = cm
# Convert the matrix into long format for ggplot
temp = as.data.frame(as.table(full_cm))
# Add model and class variable labels
temp$Model = model_name
temp$ClassVariable = "CArmedForces"
# Add this model's data to the full heatmap dataset
heatmap_data = rbind(heatmap_data, temp)
}
# Return the completed CArmedForces heatmap dataset
return(heatmap_data)
}
# Create CArmedForces heatmap data including both Decision Tree versions
heatmap_CArmedForces = make_carmedforces_heatmap_data()
# Plot confusion matrix heatmaps for CArmedForces
# The default and improved Decision Tree models are shown separately
ggplot(
heatmap_CArmedForces,
aes(x = Predicted, y = Actual, fill = Freq)
) +
geom_tile(colour = "white", linewidth = 1) +
geom_text(
aes(label = Freq),
size = 5,
fontface = "bold"
) +
facet_wrap(~ Model, nrow = 2) +
labs(
title = "Confusion Matrix Heatmaps for CArmedForces Models",
subtitle = "Default and improved Decision Tree models are shown separately",
x = "Predicted Class",
y = "Actual Class",
fill = "Count"
) +
scale_y_discrete(limits = rev) +
theme_minimal(base_size = 14) +
theme(
plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
plot.subtitle = element_text(hjust = 0.5, size = 12),
strip.text = element_text(face = "bold", size = 12),
axis.title = element_text(face = "bold"),
axis.text = element_text(face = "bold"),
legend.title = element_text(face = "bold")
)
