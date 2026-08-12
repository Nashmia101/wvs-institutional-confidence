# Q3_train_test_split

# Set seed for reproducibility using student ID
set.seed(34091904)
# Randomly sample 70% of row indices for training set
train.row = sample(1:nrow(WD_model), 0.7 * nrow(WD_model))
# Create training set using sampled row indices
WD.train = WD_model[train.row, ]
# Create test set using remaining 30% of rows
WD.test = WD_model[-train.row, ]
# Confirm dimensions of training and test sets
dim(WD.train)
dim(WD.test)
