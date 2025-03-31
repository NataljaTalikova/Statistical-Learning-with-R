# Download the libraries
library(ggplot2)
library(reshape2)
library(corrplot)


# EDA and initial Features Selection
# Read CSV file
df <- read.csv("taiw_bus_data.csv")
# cat("Original number of columns:", ncol(df), "\n")
# Check for missing values
# missing_values <- colSums(is.na(df))
# print(missing_values)

#Find correlation coefficients for all potential features
#Check standard deviation
sapply(df, sd, na.rm = TRUE)
# check the column for zero values
sum(df$Net_Income_Flag == 0)
# create a new data frame ndf without Net_Income_Flag
ndf <- df[, !names(df) %in% "Net.Income.Flag", drop = FALSE]
#check that new data frame does not have the problematic column
names(ndf)

# 1 Compute correlation matrix - did not work - go back
cor_matrix <- cor(ndf, use = "complete.obs")

# keep only the upper triangle
cor_matrix[lower.tri(cor_matrix, diag = TRUE)] <- NA  # Remove duplicates & self-correlation
cor_data <- melt(cor_matrix, na.rm = TRUE)  # Remove NA values

str(cor_data) # finds names of the cor matrix columns

# Filters and splits into two lists:
cor_remove <- cor_data[abs(cor_data$value) > 0.85, ]  # Strong correlation > 85%
cor_review <- cor_data[abs(cor_data$value) >= 0.80 & abs(cor_data$value) <= 0.85, ]  # 80-85% correlation

# Print both lists
cat(" > 85% \n")
print(cor_remove)
cat("\n  80% - 85% \n")
print(cor_review)

cor_remove <- cor_data[abs(cor_data$value) > 0.84, ] #creates list of columns to remove
columns_to_remove <- unique(cor_remove$Var2) # only unique names
ndf_cleaned <- ndf[, !names(ndf) %in% columns_to_remove]  # remove columns with highly correlated values from the df

cat("Original number of columns:", ncol(ndf), "\n")
cat("Number of columns after removal:", ncol(ndf_cleaned), "\n")

# Check distribution for the target variable
ggplot(ndf_cleaned, aes(x = as.factor(Bankrupt.), fill = as.factor(Bankrupt.))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "navy", "1" = "red")) +
  labs(title = "Target Variable(Bankrupt) Distribution",
       x = "Target Variable",
       y = "Count",
       fill = "Class") +
  theme_minimal() +  # Clean visualization
  theme(
    text = element_text(size = 14),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "top"
  )
  
# Compute class distribution
class_counts <- table(ndf_cleaned$Bankrupt.)

# Convert to percentages
class_percent <- prop.table(class_counts) * 100

# Print the results
cat("Class Distribution (%):\n")
print(round(class_percent, 2))  # Round to 2 decimal places

#visually checked the remaining data
write.csv(ndf_cleaned, "C:/Users/ntali/Desktop/Statistical Learning with R/R and data/ndf_cleaned.csv", row.names = FALSE)
summary(ndf_cleaned)

# Function to detect outlier rows (excluding specific numeric columns)
detect_outlier_rows <- function(df, exclude_columns) {
  # Ensure column names match exactly
  exclude_columns <- intersect(names(df), exclude_columns)
  
  # Select all numeric columns except the ones to exclude
  numeric_cols_to_check <- setdiff(names(df), exclude_columns)
  
  # Assertion check: These columns should NOT be in numeric_cols_to_check
  if (any(exclude_columns %in% numeric_cols_to_check)) {
    stop("Error: Excluded columns were NOT removed properly!")
  } else {
    cat("✔ Excluded columns successfully removed!\n")
  }
  
  # Compute outliers for selected numeric columns
  outlier_flags <- as.data.frame(lapply(df[, numeric_cols_to_check, drop = FALSE], function(x) {
    upper_limit <- quantile(x, 0.999, na.rm = TRUE)  # Top 1%
    lower_limit <- quantile(x, 0.001, na.rm = TRUE)  # Bottom 1%
    return(x < lower_limit | x > upper_limit)  # TRUE if outlier
  })) 
  
  # Identify rows where at least one column has an outlier
  row_outliers <- rowSums(outlier_flags) > 0  
  return(row_outliers)
}

# Define columns to exclude
exclude_columns <- c("Liability.Assets.Flag", "Bankrupt.")  # These should NOT be used for outlier detection

# Compute outliers for all columns except the excluded ones
outlier_rows <- detect_outlier_rows(ndf_cleaned, exclude_columns)

# Count how many of these outliers have target_variable = 1
outliers_with_target_1 <- sum(ndf_cleaned$Bankrupt.[outlier_rows] == 1)

# Print results
cat("Total rows with outliers (excluding the two specified columns):", sum(outlier_rows), "\n")
cat("Rows with outliers where target_variable = 1:", outliers_with_target_1, "\n")



#XGBOOST

library(xgboost)
library(caret)
library(Matrix)

set.seed(42)  # For reproducibility


target_var <- "Bankrupt." #sets target variable
feature_vars <- setdiff(names(ndf_cleaned), target_var)

#train test split
train_index <- createDataPartition(ndf_cleaned[[target_var]], p = 0.8, list = FALSE)
train_data <- ndf_cleaned[train_index, ]
test_data  <- ndf_cleaned[-train_index, ]

#train split to train and validation, used to calibrate tree depth
val_index <- createDataPartition(train_data[[target_var]], p = 0.8, list = FALSE)
train_final <- train_data[val_index, ]
val_data <- train_data[-val_index, ]

#computes class balancing weights
n_positive <- sum(train_final[[target_var]] == 1)
n_negative <- sum(train_final[[target_var]] == 0)
scale_pos_weight <- (n_negative / n_positive)
scale_pos_weight_reduced <- (n_negative / n_positive) * 1.10
cat("Scale Pos Weight:", scale_pos_weight, "\n")

# rearranges everything in XGB Matrix
dtrain <- xgb.DMatrix(data = as.matrix(train_final[, feature_vars]), label = train_final[[target_var]])
dval   <- xgb.DMatrix(data = as.matrix(val_data[, feature_vars]),   label = val_data[[target_var]])
dtest  <- xgb.DMatrix(data = as.matrix(test_data[, feature_vars]),  label = test_data[[target_var]])

depth_values <- c(4, 6, 8, 10)
cv_results <- lapply(depth_values, function(depth) {
  xgb_cv <- xgb.cv(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      scale_pos_weight = scale_pos_weight_reduced,
      eta = 0.1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      max_depth = depth
    ),
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  list(depth = depth, best_auc = max(xgb_cv$evaluation_log$test_auc_mean))
})

best_result <- cv_results[[which.max(sapply(cv_results, function(x) x$best_auc))]]
best_depth <- best_result$depth
cat("Optimal Max Depth:", best_depth, "\n")

feature_selection_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = scale_pos_weight_reduced,
  eta = 0.12,
  subsample = 0.8,
  colsample_bytree = 1.0,
  max_depth = 6,
  min_child_weight = 7,
  gamma = 1
)

feature_selection_model <- xgb.train(
  params = feature_selection_params,
  data = dtrain,               # <--- using 'dtrain'
  nrounds = 500,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 40,
  verbose = 1
)

print(feature_selection_model$best_iteration)

# Remove 'feature_names = feature_vars' to avoid mismatch
importance <- xgb.importance(model = feature_selection_model)
print(importance)

# Optionally select top features
top_features <- importance$Feature[1:25]
cat("Top 25 features:\n", top_features, "\n")


# Convert Data to XGBoost DMatrix using selected features
dtrain_selected <- xgb.DMatrix(data = as.matrix(train_final[, top_features]), label = train_final[[target_var]])
dval_selected   <- xgb.DMatrix(data = as.matrix(val_data[, top_features]), label = val_data[[target_var]])
dtest_selected  <- xgb.DMatrix(data = as.matrix(test_data[, top_features]), label = test_data[[target_var]])

# Train final model using selected features
final_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    scale_pos_weight = scale_pos_weight_reduced,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    max_depth = best_depth,
    min_child_weight = 5,  
    gamma = 1  # introduce gamma
  ),
  data = dtrain_selected,
  nrounds = 300,
  watchlist = list(train = dtrain_selected, val = dval_selected),
  early_stopping_rounds = 40,
  verbose = 1
)

# Evaluate on Test Data
pred_probs <- predict(final_model, dtest_selected)
pred_labels <- ifelse(pred_probs > 0.48, 1, 0)

conf_matrix <- confusionMatrix(as.factor(pred_labels), as.factor(test_data[[target_var]]))
print(conf_matrix)

# AUC Score
auc_score <- pROC::auc(pROC::roc(test_data[[target_var]], pred_probs))
cat("Test AUC:", auc_score, "\n")
print(final_model$best_iteration)


#SVM

library(themis) 
library(recipes)
library(e1071)
library(caret)

set.seed(42)

# Convert target variable to factor BEFORE feature selection
ndf_cleaned$Bankrupt. <- as.factor(ndf_cleaned$Bankrupt.)  

# Keep only the selected features + target variable
ndf_selected <- ndf_cleaned[, c(top_features, "Bankrupt.")]

# Train-test split (80/20)
train_index <- createDataPartition(ndf_selected$Bankrupt., p = 0.8, list = FALSE)
train_data <- ndf_selected[train_index, ]
test_data  <- ndf_selected[-train_index, ]

# Further split train into train/validation (80/20)
val_index <- createDataPartition(train_data$Bankrupt., p = 0.8, list = FALSE)
train_final <- train_data[val_index, ]
val_data <- train_data[-val_index, ]

# Prepare the data for SVM: Step SMOTE + standardize
recipe_svm <- recipe(Bankrupt. ~ ., data = train_final) %>%
  step_smote(Bankrupt., over_ratio = 1) %>%  # Apply SMOTE
  step_normalize(all_numeric_predictors())  

# Prep the recipe once and apply it to all datasets
recipe_prep <- prep(recipe_svm)

# Apply transformations
train_balanced <- bake(recipe_prep, new_data = NULL) 
val_scaled  <- bake(recipe_prep, new_data = val_data)
test_scaled <- bake(recipe_prep, new_data = test_data)

# Start the timer
start_time <- Sys.time()

# Tune SVM cost and gamma for radial kernel
tune_grid <- tune.svm(
  Bankrupt. ~ ., 
  data = train_balanced, 
  kernel = "radial", 
  cost = 2^(0:2),   # Only testing {1, 2, 4}
  gamma = 2^(-3:-1), # Only testing {0.125, 0.25, 0.5}
  tunecontrol = tune.control(sampling = "cross", cross = 3)  # 3-fold CV
)

# End the timer
end_time <- Sys.time()

# Calculate and print time taken
time_taken <- end_time - start_time
cat("SVM tuning completed in:", time_taken, "seconds\n")

# Best model
best_svm_model <- tune_grid$best.model
print(best_svm_model)

# Evaluate on Validation Set
val_preds <- predict(best_svm_model, val_scaled)
conf_matrix_val <- confusionMatrix(as.factor(val_preds), as.factor(val_scaled$Bankrupt.))
print(conf_matrix_val)
