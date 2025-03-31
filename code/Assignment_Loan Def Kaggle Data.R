# The code was created with code editor's assistance
#Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr)
library(reshape2)
library(caret)
library(smotefamily)
library(glmnet)
library(e1071)

# Check target variable class distribution before sampling
full_data <- read.csv("C:/Users/ntali/Desktop/Statistical Learning with R/R and data/Loan_default.csv")


# Plot distribution of the original Default variable
ggplot(full_data, aes(x = as.factor(Default), fill = as.factor(Default))) +
  geom_bar(width = 0.6) +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "tomato"), name = "Default") +
  labs(title = "Class Distribution in Original Dataset",
       x = "Default",
       y = "Count") +
  theme_minimal()

# Function to sample data while maintaining class distribution
sample_imbalanced_data <- function(file_path, target_var, sample_size = 0.1, random_seed = 42) {
 
  set.seed(random_seed) # for reproducibility
  data <- read.csv(file_path) # reads data from file
  
  # Stratified sampling
  sampled_data <- data %>%
    group_by(!!sym(target_var)) %>%
    sample_frac(size = sample_size)
  
  return(sampled_data)
}

file_path <- "C:/Users/ntali/Desktop/Statistical Learning with R/R and data/Loan_default.csv"   
target_var <- "Default"  
sampled_data <- sample_imbalanced_data(file_path, target_var, sample_size = 0.1)


# Saves the sampled data
write.csv(sampled_data, "C:/Users/ntali/Desktop/Statistical Learning with R/R and data/sampled_LD_Kaggle.csv", row.names = FALSE)

# Create Data frame without ID and Age columns
df <- read.csv("sampled_LD_Kaggle.csv") %>% select(-c(LoanID, Age))
ncol(df) #check
nrow(df)
anyNA(df)

#investigate type of variables for preprocessing
data.frame(Variable = names(df), Type = sapply(df, class))


# create correlation matrix for numerical columns
numeric_df <- df[sapply(df, is.numeric)]
correlation_matrix <- cor(numeric_df, use = "pairwise.complete.obs") # computes correlation matrix
melted_corr <- melt(correlation_matrix) # converts matrix to long format for ggplot2

# Heathmap
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +  # Create the heatmap
  geom_text(aes(label = round(value, 2)), size = 4, color = "midnightblue") +  # Display Pearson correlation coefficients
  scale_fill_gradient2(low = "royalblue", high = "salmon", mid = "azure", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name = "Pearson Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Pearson Correlation", x = "", y = "")

# Investigate relations between categorical variables

# 1. transform all categorical variables and the target var to factors
df$Default <- as.factor( df$Default)  # converts target var values from numerical to categorical
categorical_vars <- names(df)[sapply(df, is.factor) | sapply(df, is.character)]

# 2. create Chi-Square matrix and compute p-values for each pair of categorical variables
chi_sq_matrix <- matrix(NA, nrow = length(categorical_vars), ncol = length(categorical_vars),
                        dimnames = list(categorical_vars, categorical_vars)) # creates an empty matrix to store p-values

for (i in seq_along(categorical_vars)) {
  for (j in seq_along(categorical_vars)) {
    if (i != j) {
      chi_sq_matrix[i, j] <- chisq.test(table( df[[categorical_vars[i]]],  df[[categorical_vars[j]]]))$p.value
    } else {
      chi_sq_matrix[i, j] <- NA  # Self-comparison always gives p-value = NA
    }
  }
}

melted_chi_sq <- melt(chi_sq_matrix, na.rm = TRUE) # transforms the matrix for plotting

# 3. Plot the chi-square heatmap
ggplot(melted_chi_sq, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 3)), color = "midnightblue", size = 4) +  # Show p-values
  scale_fill_gradientn(colors = c("orangered", "azure", "mediumseagreen"), 
                       values = c(0, 0.05, 1),  # Breakpoints: red at 0, white at 0.05, blue at 1
                       limits = c(0, 1),
                       name = "p-value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Chi-Square p-Value", x = "", y = "")

#reduce insignificant columns from df
df_clean <- df[, !(names(df) %in% "LoanPurpose")]

#Box plots and outliers checks/treatments
numeric_cols <- names(df_clean)[
  sapply(df_clean, is.numeric) & names(df_clean) != "Default"
]

# Pivot data to long format for ggplot
df_long <- df_clean %>%
  select(all_of(numeric_cols)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Create boxplots in a single figure
ggplot(df_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "skyblue", outlier.color = "red", outlier.size = 1) +
  facet_wrap(~ Variable, scales = "free", ncol = 4) +
  theme_minimal() +
  labs(title = "Boxplots of Numeric Variables", x = NULL, y = NULL) +
  theme(axis.text.x = element_blank(),
        strip.text = element_text(size = 10))
outlier_summary <- sapply(df_clean[numeric_cols], function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))
})

print(outlier_summary)

# One hot encoding
df_clean$Default <- as.integer(as.character(df_clean$Default))  # converts and assign target back to integer
formula_one_hot <- as.formula("~ . - Default - 1") # to transform all but D.
df_encoded <- model.matrix(formula_one_hot, data = df_clean )
df_encoded <- cbind(df_encoded, Default = df_clean$Default) # add D. back to df_encoded
df_encoded <- as.data.frame(df_encoded) # convert back to Data Frame

dim(df_encoded)
head(df_encoded)

round(sum(df_encoded$Default == 1) / sum(df_encoded$Default == 0), 3) # proportion of the default to sample


#STRATIFIED SPLIT TRAIN - TEST
set.seed(42)

train_index <- createDataPartition(df_encoded$Default, p = 0.8, list = FALSE)
train_final <- df_encoded[train_index, ]    # Used for training (with SMOTE, scaling, etc.)
test_data   <- df_encoded[-train_index, ]   # Used only for final evaluation

# Ensure 'Default' is a factor for classification
train_final$Default <- factor(train_final$Default, levels = c("0", "1"))
test_data$Default   <- factor(test_data$Default, levels = c("0", "1"))


#SMOTE NO SCALING
train_final$Default <- as.numeric(as.character(train_final$Default))  # required for SMOTE
smote_out <- SMOTE(
  X = train_final[, -which(names(train_final) == "Default")],
  target = train_final$Default,
  K = 5
)

train_smote <- smote_out$data
train_smote$Default <- factor(train_smote$class)
train_smote$class <- NULL

# Fit log reg + predict
logit_model <- glm(Default ~ ., data = train_smote, family = binomial)
test_data$Default <- factor(test_data$Default)  # if not already
logit_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_preds <- ifelse(logit_probs >= 0.55, "1", "0") |> factor(levels = c("0", "1"))

confusionMatrix(logit_preds, test_data$Default, positive = "1") # evaluate

#SMOTE SCALING NUMERIC, LASSO REGULARIZATION
# 1. Scale numeric
numeric_cols <- names(train_smote)[sapply(train_smote, is.numeric) & names(train_smote) != "Default"]
scaler <- preProcess(train_smote[, numeric_cols], method = c("center", "scale"))
train_smote[, numeric_cols] <- predict(scaler, train_smote[, numeric_cols])
test_data[, numeric_cols]   <- predict(scaler, test_data[, numeric_cols])

# 2. Prepare design matrices for glmnet
X_train <- as.matrix(train_smote[, setdiff(names(train_smote), "Default")])
y_train <- as.numeric(as.character(train_smote$Default))
X_test <- as.matrix(test_data[, setdiff(names(test_data), "Default")])
y_test <- as.numeric(as.character(test_data$Default))

# 3. Fit cross-validated Lasso model
set.seed(42)
cv_model <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "binomial",
  alpha = 1,           # Lasso
  type.measure = "class"  # Or "auc" / "deviance"
)

# 4. Predict on test set using best lambda
test_probs <- predict(cv_model, newx = X_test, s = "lambda.min", type = "response")

# 5. Thresholding and evaluation
threshold <- 0.55  # or try 0.5, 0.3, etc.
test_preds <- ifelse(test_probs >= threshold, 1, 0) |> factor(levels = c(0, 1))
confusionMatrix(test_preds, factor(y_test, levels = c(0, 1)), positive = "1")


# SVM
# uses same stratified train-val -test split and SMOTE augmentation as Reg Log Reg
# RBF KERNEL
# 1. Fit and evaluate the base model
svm_model <- svm(
  Default ~ .,
  data = train_smote,
  kernel = "radial",
  cost = 1,     # Try tuning this later
  gamma = 0.1,  # Or "auto" / tune later
  probability = TRUE
)

svm_probs <- attr(predict(svm_model, test_data, probability = TRUE), "probabilities")[, "1"]

threshold <- 0.5 # The custom threshold 
svm_preds <- ifelse(svm_probs >= threshold, "1", "0") |> factor(levels = c("0", "1"))

# Evaluate
confusionMatrix(svm_preds, test_data$Default, positive = "1")

# OMMIT TUNING CODE READING - go to tuned_svm <- readRDS("tuned_svm_model.rds")
# 2. Access training time through the tentative model
set.seed(42)
system.time({
  tuned_svm <- tune(
    svm,
    Default ~ .,
    data = train_smote,
    kernel = "radial",
    ranges = list(
      cost = c(1, 10),
      gamma = c(0.01, 0.1)
    ),
    tunecontrol = tune.control(cross = 5)
  )
})

# 3. RBF TUNING 
set.seed(42)
system.time({
  tuned_svm <- tune(
    svm,
    Default ~ .,
    data = train_smote,
    kernel = "radial",
    ranges = list(
      cost = c(0.1, 1, 10, 100),
      gamma = c(0.001, 0.005, 0.01, 0.05, 0.1)
    ),
    tunecontrol = tune.control(cross = 5),
    probability = TRUE
  )
})

saveRDS(tuned_svm, file = "tuned_svm_RBF_model.rds") # saves the tuned model

#START READING CODE HERE
tuned_svm <- readRDS("tuned_svm_RBF_model.rds") # Reload model
best_svm <- tuned_svm$best.model

# Predicts probabilities on test set
svm_RBF_probs <- attr(predict(best_svm, test_data, probability = TRUE), "probabilities")[, "1"]

# Search for best threshold to maximize F1-score
thresholds <- seq(0.2, 0.7, by = 0.01)
f1_scores <- sapply(thresholds, function(th) {
  preds <- ifelse(svm_RBF_probs >= th, "1", "0")
  preds <- factor(preds, levels = c("0", "1"))
  cm <- confusionMatrix(preds, test_data$Default, positive = "1")
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  if ((precision + recall) == 0) return(0)
  return(2 * precision * recall / (precision + recall))  # F1
})

# Best threshold
best_thresh <- thresholds[which.max(f1_scores)]
cat("Best threshold (F1):", best_thresh, "\n")

# evaluate with best threshold
svm_RBF_final_preds <- ifelse(svm_RBF_probs >= best_thresh, "1", "0") |> factor(levels = c("0", "1"))
confusionMatrix(svm_RBF_final_preds, test_data$Default, positive = "1")

#SVM  RBFKERNEL UNIT 
svm_model <- svm(
  Default ~ .,
  data = train_smote,
  kernel = "radial",
  cost = 1,     # Try tuning this later
  gamma = 0.1,  # Or "auto" / tune later
  probability = TRUE
)

# Step 4: Predict probabilities on test set
svm_probs <- attr(predict(svm_model, test_data, probability = TRUE), "probabilities")[, "1"]

# Step 5: Predict with custom threshold
threshold <- 0.5
svm_preds <- ifelse(svm_probs >= threshold, "1", "0") |> factor(levels = c("0", "1"))

# Step 6: Evaluate
confusionMatrix(svm_preds, test_data$Default, positive = "1")

#SVM  POLYNOMIAL KERNEL
set.seed(42)
svm_poly <- svm(
  Default ~ .,
  data = train_smote,
  kernel = "polynomial",
  degree = 2,          # Try 2 or 3
  cost = 10,
  coef0 = 1,           # Can tune this too
  probability = TRUE
)

# Predict + Evaluate
svm_poly_probs <- attr(predict(svm_poly, test_data, probability = TRUE), "probabilities")[, "1"]
svm_poly_preds <- ifelse(svm_poly_probs >= 0.5, "1", "0") |> factor(levels = c("0", "1"))

confusionMatrix(svm_poly_preds, test_data$Default, positive = "1")

# The first SVM RBF result
# Reload model
tuned_svm <- readRDS("x tuned_svm_model.rds")
best_svm <- tuned_svm$best.model

# Predict classes (default 0.5 boundary)
svm_preds <- predict(best_svm, test_data)

# Ensure correct format
svm_preds <- factor(svm_preds, levels = c("0", "1"))
test_data$Default <- factor(test_data$Default, levels = c("0", "1"))

# Confusion matrix with default threshold
confusionMatrix(svm_preds, test_data$Default, positive = "1")