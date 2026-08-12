# Q4_classification_models

# Initialise empty lists to store decision tree models, predictions, and confusion matrices
tree_models = list()
tree_predictions = list()
tree_confusion_matrices = list()
# Loop through each class variable and fit a separate decision tree model
for (target in class_vars) {
# Identify the other two class variables to exclude from predictors
other_class_vars = class_vars[class_vars != target]
# Build model formula excluding Country, Wave, and other class variables
formula_text = paste(
target,
"~ . - Country - Wave -",
paste(other_class_vars, collapse = " - ")
)
model_formula = as.formula(formula_text)
# Fit decision tree using default tree settings
tree_models[[target]] = tree(
model_formula,
data = WD.train
)
# Print model summary
print(summary(tree_models[[target]]))
# Predict class labels on test data
tree_predictions[[target]] = predict(
tree_models[[target]],
WD.test,
type = "class"
)
# Confusion matrix comparing actual vs predicted class labels
tree_confusion_matrices[[target]] = table(
actual = WD.test[[target]],
predicted = tree_predictions[[target]]
)
print(tree_confusion_matrices[[target]])
}
# Plot decision tree for CCivilService
plot(tree_models$CCivilService)
text(tree_models$CCivilService, pretty = 0)
# Plot decision tree for CChurches
plot(tree_models$CChurches)
text(tree_models$CChurches, pretty = 0)
# Plot decision tree for CArmedForces
plot(tree_models$CArmedForces)
text(tree_models$CArmedForces, pretty = 0)
# Ensure CArmedForces is stored as a factor with levels 0 and 1 in both train and test sets
WD.train$CArmedForces = factor(WD.train$CArmedForces, levels = c(0, 1))
WD.test$CArmedForces = factor(WD.test$CArmedForces, levels = c(0, 1))
# Fit improved decision tree for CArmedForces using same train/test split
# Equal class priors applied so both classes are treated as equally important
# Tree complexity controlled to allow meaningful splits while reducing overfitting risk
tree_CArmedForces_improved = rpart(
CArmedForces ~ . - Country - Wave - CCivilService - CChurches,
data = WD.train,
method = "class",
parms = list(prior = c(0.5, 0.5)),
control = rpart.control(
cp = 0.001, # minimum improvement required for a split
minsplit = 40, # minimum observations required before a node can split
minbucket = 15, # minimum observations required in each terminal node
maxdepth = 5, # maximum tree depth to prevent overfitting
xval = 10 # 10-fold cross-validation
)
)
print(tree_CArmedForces_improved)
summary(tree_CArmedForces_improved)
# Plot improved CArmedForces decision tree
rpart.plot(
tree_CArmedForces_improved,
type = 2,
extra = 104,
fallen.leaves = TRUE,
main = "Improved Decision Tree for CArmedForces"
)
# Predict on test data using improved CArmedForces tree
pred_CArmedForces_improved = predict(
tree_CArmedForces_improved,
newdata = WD.test,
type = "class"
)
# Confusion matrix for improved CArmedForces tree
cm_CArmedForces_improved = table(
actual = WD.test$CArmedForces,
predicted = pred_CArmedForces_improved
)
cm_CArmedForces_improved
# Compute performance metrics for improved CArmedForces tree
TP = cm_CArmedForces_improved["1", "1"]
TN = cm_CArmedForces_improved["0", "0"]
FP = cm_CArmedForces_improved["0", "1"]
FN = cm_CArmedForces_improved["1", "0"]
accuracy = (TP + TN) / sum(cm_CArmedForces_improved)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))
# Store metrics in a dataframe
CArmedForces_improved_metrics = data.frame(
Model = "Improved Decision Tree",
ClassVariable = "CArmedForces",
Accuracy = round(accuracy, 4),
Sensitivity = round(sensitivity, 4),
Specificity = round(specificity, 4),
Precision = round(precision, 4),
F1_Score = round(f1_score, 4)
)
CArmedForces_improved_metrics
# Print cross-validation results and plot cross-validation error by tree size
printcp(tree_CArmedForces_improved)
plotcp(tree_CArmedForces_improved)
# Initialise empty lists to store Naive Bayes models, predictions, and confusion matrices
nb_models = list()
nb_predictions = list()
nb_confusion_matrices = list()
# Loop through each class variable and fit a separate Naive Bayes model
for (target in class_vars) {
# Identify the other two class variables to exclude from predictors
other_class_vars = class_vars[class_vars != target]
# Build model formula excluding Country, Wave, and other class variables
formula_text = paste(
target,
"~ . - Country - Wave -",
paste(other_class_vars, collapse = " - ")
)
model_formula = as.formula(formula_text)
# Fit Naive Bayes model using default settings with no Laplace smoothing
nb_models[[target]] = naiveBayes(
model_formula,
data = WD.train
)
print(nb_models[[target]])
# Predict class labels on test data
nb_predictions[[target]] = predict(
nb_models[[target]],
WD.test,
type = "class"
)
# Confusion matrix comparing actual vs predicted class labels
nb_confusion_matrices[[target]] = table(
actual = WD.test[[target]],
predicted = nb_predictions[[target]]
)
print(nb_confusion_matrices[[target]])
}
# Initialise empty lists to store Bagging models, predictions, and confusion matrices
bagging_models = list()
bagging_predictions = list()
bagging_confusion_matrices = list()
# Loop through each class variable and fit a separate Bagging model
for (target in class_vars) {
# Identify the other two class variables to exclude from predictors
other_class_vars = class_vars[class_vars != target]
# Build model formula excluding Country, Wave, and other class variables
formula_text = paste(
target,
"~ . - Country - Wave -",
paste(other_class_vars, collapse = " - ")
)
model_formula = as.formula(formula_text)
set.seed(34091904)
# Fit Bagging model using default settings (mfinal = 100 bootstrap iterations)
bagging_models[[target]] = bagging(
model_formula,
data = WD.train
)
# Print variable importance from Bagging model
print(bagging_models[[target]]$importance)
# Predict on test data and extract predicted class labels
bagging_pred_object = predict(
bagging_models[[target]],
newdata = WD.test
)
bagging_predictions[[target]] = as.factor(bagging_pred_object$class)
# Confusion matrix comparing actual vs predicted class labels
bagging_confusion_matrices[[target]] = table(
actual = WD.test[[target]],
predicted = bagging_predictions[[target]]
)
print(bagging_confusion_matrices[[target]])
}
# Initialise empty lists to store Boosting models, predictions, and confusion matrices
boosting_models = list()
boosting_predictions = list()
boosting_confusion_matrices = list()
# Loop through each class variable and fit a separate Boosting model
for (target in class_vars) {
# Identify the other two class variables to exclude from predictors
other_class_vars = class_vars[class_vars != target]
# Build model formula excluding Country, Wave, and other class variables
formula_text = paste(
target,
"~ . - Country - Wave -",
paste(other_class_vars, collapse = " - ")
)
model_formula = as.formula(formula_text)
set.seed(34091904)
# Fit Boosting model using default AdaBoost settings (mfinal = 100 iterations)
boosting_models[[target]] = boosting(
model_formula,
data = WD.train
)
# Print variable importance from Boosting model
print(boosting_models[[target]]$importance)
# Predict on test data and extract predicted class labels
boosting_pred_object = predict(
boosting_models[[target]],
newdata = WD.test
)
boosting_predictions[[target]] = as.factor(boosting_pred_object$class)
# Confusion matrix comparing actual vs predicted class labels
boosting_confusion_matrices[[target]] = table(
actual = WD.test[[target]],
predicted = boosting_predictions[[target]]
)
print(boosting_confusion_matrices[[target]])
}
# Initialise empty lists to store Random Forest models, predictions, and confusion matrices
rf_models = list()
rf_predictions = list()
rf_confusion_matrices = list()
# Loop through each class variable and fit a separate Random Forest model
for (target in class_vars) {
# Identify the other two class variables to exclude from predictors
other_class_vars = class_vars[class_vars != target]
# Build model formula excluding Country, Wave, and other class variables
formula_text = paste(
target,
"~ . - Country - Wave -",
paste(other_class_vars, collapse = " - ")
)
model_formula = as.formula(formula_text)
set.seed(34091904)
# Fit Random Forest model using default settings (ntree = 500)
# importance = TRUE enables variable importance extraction
rf_models[[target]] = randomForest(
model_formula,
data = WD.train,
importance = TRUE
)
print(rf_models[[target]])
# Predict class labels on test data
rf_predictions[[target]] = predict(
rf_models[[target]],
WD.test,
type = "class"
)
# Confusion matrix comparing actual vs predicted class labels
rf_confusion_matrices[[target]] = table(
actual = WD.test[[target]],
predicted = rf_predictions[[target]]
)
print(rf_confusion_matrices[[target]])
}
