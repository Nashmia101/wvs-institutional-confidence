# Q7_model_comparison

# Add class variable label to each Q5 performance table for merging
performance_CCivilService = performance_tables$CCivilService
performance_CCivilService$ClassVariable = "CCivilService"
performance_CChurches = performance_tables$CChurches
performance_CChurches$ClassVariable = "CChurches"
performance_CArmedForces = performance_tables$CArmedForces
performance_CArmedForces$ClassVariable = "CArmedForces"
# Combine all three class variable performance tables into one dataframe
all_performance_results = rbind(
performance_CCivilService,
performance_CChurches,
performance_CArmedForces
)
# Reorder columns for readability
all_performance_results = all_performance_results[, c(
"ClassVariable", "Model", "Accuracy", "Precision", "Recall", "F1_Score"
)]
all_performance_results
# Combine Q6 AUC tables for all three class variables into one dataframe
all_auc_results = rbind(
auc_tables$CCivilService,
auc_tables$CChurches,
auc_tables$CArmedForces
)
all_auc_results
# Merge Q5 performance metrics with Q6 AUC values by class variable and model name
all_results = merge(
all_performance_results,
all_auc_results,
by = c("ClassVariable", "Model")
)
all_results
# Compute average performance metrics per model across all three class variables
# Overall_Average is the mean of all five metrics for a single overall ranking score
average_model_comparison = all_results %>%
group_by(Model) %>%
summarise(
Average_Accuracy = round(mean(Accuracy), 4),
Average_Precision = round(mean(Precision), 4),
Average_Recall = round(mean(Recall), 4),
Average_F1_Score = round(mean(F1_Score), 4),
Average_AUC = round(mean(AUC), 4),
Overall_Average = round(mean(c(Accuracy, Precision, Recall, F1_Score, AUC)), 4)
) %>%
arrange(desc(Overall_Average))
# Print average model comparison table sorted by overall average descending
average_model_comparison
