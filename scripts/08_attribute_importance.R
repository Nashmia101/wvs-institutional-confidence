# Q8_attribute_importance

# Extract variable importance values from Bagging, Boosting, and Random Forest models
# for a given class variable target
extract_importance = function(target) {
# Extract Bagging importance
bag_imp = data.frame(
Variable = names(bagging_models[[target]]$importance),
Bagging = as.numeric(bagging_models[[target]]$importance)
)
# Extract Boosting importance
boost_imp = data.frame(
Variable = names(boosting_models[[target]]$importance),
Boosting = as.numeric(boosting_models[[target]]$importance)
)
# Extract Random Forest importance using MeanDecreaseGini if available
rf_imp_matrix = importance(rf_models[[target]])
if ("MeanDecreaseGini" %in% colnames(rf_imp_matrix)) {
rf_values = rf_imp_matrix[, "MeanDecreaseGini"]
} else {
rf_values = rf_imp_matrix[, 1]
}
rf_imp = data.frame(
Variable = rownames(rf_imp_matrix),
Random_Forest = as.numeric(rf_values)
)
# Merge all three importance tables by variable name
importance_table = merge(bag_imp, boost_imp, by = "Variable", all = TRUE)
importance_table = merge(importance_table, rf_imp, by = "Variable", all = TRUE)
# Replace any missing values with zero
importance_table[is.na(importance_table)] = 0
return(importance_table)
}
# Extract importance for each class variable
importance_CCivilService = extract_importance("CCivilService")
importance_CChurches = extract_importance("CChurches")
importance_CArmedForces = extract_importance("CArmedForces")
# Normalise importance values within each model so they are comparable on a 0 to 1 scale
# Compute overall normalised importance as the average across the three models
# Sort variables by overall normalised importance descending
make_importance_data = function(importance_data) {
table_data = importance_data
# Normalise Bagging importance relative to its maximum value
if (max(table_data$Bagging) > 0) {
table_data$Bagging_Normalised = table_data$Bagging / max(table_data$Bagging)
} else {
table_data$Bagging_Normalised = 0
}
# Normalise Boosting importance relative to its maximum value
if (max(table_data$Boosting) > 0) {
table_data$Boosting_Normalised = table_data$Boosting / max(table_data$Boosting)
} else {
table_data$Boosting_Normalised = 0
}
# Normalise Random Forest importance relative to its maximum value
if (max(table_data$Random_Forest) > 0) {
table_data$Random_Forest_Normalised = table_data$Random_Forest /
max(table_data$Random_Forest)
} else {
table_data$Random_Forest_Normalised = 0
}
# Compute overall normalised importance as average across the three models
table_data = table_data %>%
mutate(
Overall_Normalised_Importance = (
Bagging_Normalised +
Boosting_Normalised +
Random_Forest_Normalised
) / 3
) %>%
arrange(desc(Overall_Normalised_Importance))
return(table_data)
}
# Prepare ranked importance data for each class variable
plot_data_CCivilService = make_importance_data(importance_CCivilService)
plot_data_CChurches = make_importance_data(importance_CChurches)
plot_data_CArmedForces = make_importance_data(importance_CArmedForces)
# Function to create faceted vertical bar charts showing attribute importance
# for each of the three ensemble models for a given class variable
plot_importance_chart = function(data, target_title) {
# Reshape data from wide to long format for faceted plotting
long_data = data %>%
select(Variable, Bagging, Boosting, Random_Forest) %>%
pivot_longer(
cols = c(Bagging, Boosting, Random_Forest),
names_to = "Model",
values_to = "Importance"
)
# Set factor levels for model names for consistent ordering in facets
long_data$Model = factor(
long_data$Model,
levels = c("Bagging", "Boosting", "Random_Forest"),
labels = c("Bagging", "Boosting", "Random Forest")
)
# Preserve variable ordering based on overall normalised importance
long_data$Variable = factor(
long_data$Variable,
levels = data$Variable
)
# Build faceted bar chart with one panel per ensemble model
ggplot(long_data, aes(x = Variable, y = Importance, fill = Model)) +
geom_col(width = 0.75, show.legend = FALSE) +
# Add importance value labels above each bar
geom_text(
aes(label = round(Importance, 2)),
vjust = -0.25,
size = 3.6,
fontface = "bold"
) +
# One row per model with free y-axis scale and shared x-axis labels
facet_wrap(
~ Model,
ncol = 1,
scales = "free_y",
axes = "all_x",
axis.labels = "all_x"
) +
# Add padding above bars so labels are not clipped
scale_y_continuous(
expand = expansion(mult = c(0, 0.15))
) +
# Wrap long variable names for readability on x-axis
scale_x_discrete(
labels = function(x) stringr::str_wrap(x, width = 8)
) +
labs(
title = paste("Attribute Importance for Predicting Confidence in", target_title),
x = "Predictor",
y = "Importance Value"
) +
theme_minimal() +
theme(
plot.title = element_text(size = 22, face = "bold", hjust = 0.5),
strip.text = element_text(size = 16, face = "bold"),
axis.title.x = element_text(size = 16, face = "bold", margin = ggplot2::margin(t = 10)),
axis.title.y = element_text(size = 16, face = "bold"),
axis.text.x = element_text(size = 10, face = "bold", angle = 45, hjust = 1, vjust = 1),
axis.text.y = element_text(size = 12, face = "bold"),
panel.spacing = grid::unit(1.2, "lines"),
plot.margin = ggplot2::margin(10, 15, 20, 15)
)
}
# Generate attribute importance charts for each class variable
figure15_CCivilService = plot_importance_chart(plot_data_CCivilService, "Civil Service")
figure16_CChurches = plot_importance_chart(plot_data_CChurches, "Churches")
figure17_CArmedForces = plot_importance_chart(plot_data_CArmedForces, "Armed
Forces")
# Display figures
figure15_CCivilService
figure16_CChurches
figure17_CArmedForces
# Save Figure 15: Attribute importance for CCivilService
ggsave(
filename = "Figure15_Attribute_Importance_CCivilService.png",
plot = figure15_CCivilService,
width = 18,
height = 12,
units = "in",
dpi = 300
)
# Save Figure 16: Attribute importance for CChurches
ggsave(
filename = "Figure16_Attribute_Importance_CChurches.png",
plot = figure16_CChurches,
width = 18,
height = 12,
units = "in",
dpi = 300
)
# Save Figure 17: Attribute importance for CArmedForces
ggsave(
filename = "Figure17_Attribute_Importance_CArmedForces.png",
plot = figure17_CArmedForces,
width = 18,
height = 12,
units = "in",
dpi = 300
)
