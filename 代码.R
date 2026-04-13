# ============================================
# 实验四：集成学习二 - Titanic数据实验
# R版本：4.4.1
# ============================================

# ---------- 1. 加载所需包 ----------
required_packages <- c("tidyverse", "caret", "randomForest", 
                       "gbm", "xgboost", "ada", "pROC", "ggplot2",
                       "corrplot", "mice", "gridExtra")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ---------- 2. 数据读取 ----------
# 从Kaggle下载Titanic数据集并放置在指定路径
train <- read.csv("C:/Users/86152/Desktop/机器学习汇报/train.csv", header = TRUE, stringsAsFactors = FALSE)
test <- read.csv("C:/Users/86152/Desktop/机器学习汇报/test.csv", header = TRUE, stringsAsFactors = FALSE)

# 查看数据结构
str(train)
summary(train)

# ---------- 3. 探索性分析 ----------
# 3.1 生存率整体分布
survive_table <- table(train$Survived)
prop.table(survive_table)
ggplot(train, aes(x = factor(Survived), fill = factor(Survived))) +
  geom_bar() +
  scale_fill_manual(values = c("#E69F00", "#56B4E9"), 
                    labels = c("死亡", "生存")) +
  labs(title = "Titanic乘客生存情况分布", x = "是否生存", y = "人数") +
  theme_minimal()

# 3.2 性别与生存率
ggplot(train, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#E69F00", "#56B4E9"), 
                    labels = c("死亡", "生存")) +
  labs(title = "性别与生存率关系", x = "性别", y = "比例") +
  theme_minimal()

# 3.3 舱位等级与生存率
ggplot(train, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#E69F00", "#56B4E9"), 
                    labels = c("死亡", "生存")) +
  labs(title = "舱位等级与生存率关系", x = "舱位等级", y = "比例") +
  theme_minimal()

# 3.4 年龄分布
ggplot(train, aes(x = Age)) +
  geom_histogram(bins = 30, fill = "#56B4E9", alpha = 0.7) +
  labs(title = "乘客年龄分布", x = "年龄", y = "频数") +
  theme_minimal()

# 3.5 缺失值分析
missing_train <- sapply(train, function(x) sum(is.na(x)))
missing_test <- sapply(test, function(x) sum(is.na(x)))
print("训练集缺失值统计：")
print(missing_train[missing_train > 0])
print("测试集缺失值统计：")
print(missing_test[missing_test > 0])

# ---------- 4. 数据预处理 ----------
# 合并训练集和测试集以便统一处理
test$Survived <- NA
full_data <- rbind(train, test)

# 4.1 处理Embarked缺失值（使用众数填充）
full_data$Embarked[full_data$Embarked == ""] <- NA
embarked_mode <- names(sort(table(full_data$Embarked), decreasing = TRUE))[1]
full_data$Embarked[is.na(full_data$Embarked)] <- embarked_mode

# 4.2 处理Fare缺失值（使用中位数填充）
full_data$Fare[is.na(full_data$Fare)] <- median(full_data$Fare, na.rm = TRUE)

# 4.3 处理Age缺失值（按Pclass分组的中位数填充）
full_data <- full_data %>%
  group_by(Pclass) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

# 4.4 删除Cabin变量（缺失率过高）
full_data$Cabin <- NULL

# 4.5 年龄分箱
full_data$AgeGroup <- cut(full_data$Age,
                          breaks = c(0, 12, 24, 59, 100),
                          labels = c("Child", "Youth", "Adult", "Elder"))

# 4.6 票价分箱
full_data$FareGroup <- cut(full_data$Fare,
                           breaks = quantile(full_data$Fare, probs = c(0, 0.33, 0.67, 1), 
                                             na.rm = TRUE),
                           labels = c("Low", "Medium", "High"),
                           include.lowest = TRUE)

# ---------- 5. 特征工程 ----------
# 5.1 家庭规模特征
full_data$FamilySize <- full_data$SibSp + full_data$Parch + 1

# 5.2 提取姓名头衔
full_data$Title <- gsub("(.*, )|(\\..*)", "", full_data$Name)
# 合并稀有头衔
rare_titles <- c("Lady", "Countess", "Capt", "Col", "Don", 
                 "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona")
full_data$Title[full_data$Title %in% rare_titles] <- "Rare"
full_data$Title[full_data$Title %in% c("Mlle", "Ms")] <- "Miss"
full_data$Title[full_data$Title == "Mme"] <- "Mrs"

# 5.3 转换为因子变量
factor_vars <- c("Survived", "Pclass", "Sex", "Embarked", 
                 "AgeGroup", "FareGroup", "Title")
for (var in factor_vars) {
  full_data[[var]] <- as.factor(full_data[[var]])
}

# ---------- 6. 准备建模数据 ----------
# 选择建模特征
feature_vars <- c("Pclass", "Sex", "AgeGroup", "FareGroup", 
                  "Embarked", "FamilySize", "Title")
# 分离训练集和测试集
train_processed <- full_data[1:nrow(train), c(feature_vars, "Survived")]
test_processed <- full_data[(nrow(train)+1):nrow(full_data), feature_vars]

# 划分训练集和验证集
set.seed(2024)
train_index <- createDataPartition(train_processed$Survived, p = 0.8, list = FALSE)
train_data <- train_processed[train_index, ]
valid_data <- train_processed[-train_index, ]

# 准备XGBoost专用矩阵格式
x_train <- model.matrix(~ . - 1, train_data[, feature_vars])
y_train <- as.numeric(train_data$Survived) - 1
x_valid <- model.matrix(~ . - 1, valid_data[, feature_vars])
y_valid <- as.numeric(valid_data$Survived) - 1

dtrain <- xgb.DMatrix(x_train, label = y_train)
dvalid <- xgb.DMatrix(x_valid, label = y_valid)

# ---------- 7. 评价函数 ----------
calculate_metrics <- function(actual, predicted, prob = NULL) {
  # 混淆矩阵
  cm <- confusionMatrix(factor(predicted), factor(actual), positive = "1")
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  
  # AUC
  if (!is.null(prob)) {
    roc_obj <- roc(actual, prob)
    auc_value <- auc(roc_obj)
  } else {
    auc_value <- NA
  }
  
  return(c(Precision = precision, Recall = recall, F1 = f1, AUC = auc_value))
}

# ---------- 8. 模型1: AdaBoost ----------
cat("\n========== 模型1: AdaBoost ==========\n")
set.seed(2024)
ada_grid <- expand.grid(
  iter = c(50, 100, 200),
  maxdepth = c(1, 3, 5),
  nu = 0.1
)

ada_best_f1 <- 0
ada_best_params <- NULL

for (i in 1:nrow(ada_grid)) {
  tryCatch({
    ada_model <- ada(Survived ~ ., data = train_data,
                     iter = ada_grid$iter[i],
                     maxdepth = ada_grid$maxdepth[i],
                     nu = ada_grid$nu[i])
    ada_pred <- predict(ada_model, valid_data, type = "vector")
    ada_prob <- predict(ada_model, valid_data, type = "prob")[, 2]
    
    metrics <- calculate_metrics(valid_data$Survived, ada_pred, ada_prob)
    
    if (metrics["F1"] > ada_best_f1) {
      ada_best_f1 <- metrics["F1"]
      ada_best_params <- ada_grid[i, ]
      ada_best_metrics <- metrics
    }
  }, error = function(e) {
    cat("AdaBoost参数组合失败:", e$message, "\n")
  })
}

cat("AdaBoost最优参数:\n")
print(ada_best_params)
cat("AdaBoost最优性能:\n")
print(round(ada_best_metrics, 4))

# ---------- 9. 模型2: GBDT ----------
cat("\n========== 模型2: GBDT ==========\n")
set.seed(2024)
gbm_grid <- expand.grid(
  n.trees = c(100, 200, 500),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

gbm_best_f1 <- 0
gbm_best_params <- NULL

for (i in 1:nrow(gbm_grid)) {
  tryCatch({
    gbm_model <- gbm(Survived ~ ., data = train_data,
                     distribution = "bernoulli",
                     n.trees = gbm_grid$n.trees[i],
                     interaction.depth = gbm_grid$interaction.depth[i],
                     shrinkage = gbm_grid$shrinkage[i],
                     n.minobsinnode = gbm_grid$n.minobsinnode[i],
                     cv.folds = 5,
                     verbose = FALSE)
    
    best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
    gbm_prob <- predict(gbm_model, valid_data, n.trees = best_iter, type = "response")
    gbm_pred <- ifelse(gbm_prob > 0.5, 1, 0)
    
    metrics <- calculate_metrics(valid_data$Survived, gbm_pred, gbm_prob)
    
    if (metrics["F1"] > gbm_best_f1) {
      gbm_best_f1 <- metrics["F1"]
      gbm_best_params <- gbm_grid[i, ]
      gbm_best_metrics <- metrics
    }
  }, error = function(e) {
    cat("GBDT参数组合失败:", e$message, "\n")
  })
}

cat("GBDT最优参数:\n")
print(gbm_best_params)
cat("GBDT最优性能:\n")
print(round(gbm_best_metrics, 4))

# ---------- 10. 模型3: XGBoost ----------
cat("\n========== 模型3: XGBoost ==========\n")
set.seed(2024)
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 5, 7),
  eta = c(0.05, 0.1, 0.3),
  gamma = c(0, 0.1, 0.2),
  subsample = c(0.7, 0.8, 1.0),
  colsample_bytree = c(0.7, 0.8, 1.0),
  min_child_weight = 1
)

xgb_best_f1 <- 0
xgb_best_params <- NULL

# 由于网格搜索组合较多，这里展示主要参数组合
xgb_params_list <- list(
  list(max_depth = 3, eta = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
  list(max_depth = 5, eta = 0.1, gamma = 0.1, subsample = 0.8, colsample_bytree = 0.8),
  list(max_depth = 5, eta = 0.05, gamma = 0, subsample = 0.7, colsample_bytree = 0.7),
  list(max_depth = 7, eta = 0.1, gamma = 0.2, subsample = 1.0, colsample_bytree = 1.0),
  list(max_depth = 3, eta = 0.3, gamma = 0, subsample = 0.8, colsample_bytree = 0.8)
)

for (params in xgb_params_list) {
  tryCatch({
    xgb_model <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = params$max_depth,
        eta = params$eta,
        gamma = params$gamma,
        subsample = params$subsample,
        colsample_bytree = params$colsample_bytree
      ),
      data = dtrain,
      nrounds = 200,
      watchlist = list(train = dtrain, valid = dvalid),
      early_stopping_rounds = 20,
      verbose = 0
    )
    
    xgb_prob <- predict(xgb_model, x_valid)
    xgb_pred <- ifelse(xgb_prob > 0.5, 1, 0)
    
    metrics <- calculate_metrics(y_valid, xgb_pred, xgb_prob)
    
    if (metrics["F1"] > xgb_best_f1) {
      xgb_best_f1 <- metrics["F1"]
      xgb_best_params <- params
      xgb_best_metrics <- metrics
    }
  }, error = function(e) {
    cat("XGBoost参数组合失败:", e$message, "\n")
  })
}

cat("XGBoost最优参数:\n")
print(xgb_best_params)
cat("XGBoost最优性能:\n")
print(round(xgb_best_metrics, 4))

# 特征重要性
importance_matrix <- xgb.importance(feature_names = feature_vars, model = xgb_model)
xgb.plot.importance(importance_matrix, main = "XGBoost特征重要性")

# ---------- 11. 模型4: RandomForest ----------
cat("\n========== 模型4: RandomForest ==========\n")
set.seed(2024)
rf_grid <- expand.grid(
  ntree = c(100, 200, 500),
  mtry = c(2, 3, 4, 5)
)

rf_best_f1 <- 0
rf_best_params <- NULL

for (i in 1:nrow(rf_grid)) {
  tryCatch({
    rf_model <- randomForest(Survived ~ ., data = train_data,
                             ntree = rf_grid$ntree[i],
                             mtry = rf_grid$mtry[i],
                             importance = TRUE)
    rf_pred <- predict(rf_model, valid_data, type = "class")
    rf_prob <- predict(rf_model, valid_data, type = "prob")[, 2]
    
    metrics <- calculate_metrics(valid_data$Survived, rf_pred, rf_prob)
    
    if (metrics["F1"] > rf_best_f1) {
      rf_best_f1 <- metrics["F1"]
      rf_best_params <- rf_grid[i, ]
      rf_best_metrics <- metrics
    }
  }, error = function(e) {
    cat("RandomForest参数组合失败:", e$message, "\n")
  })
}

cat("RandomForest最优参数:\n")
print(rf_best_params)
cat("RandomForest最优性能:\n")
print(round(rf_best_metrics, 4))

# 随机森林特征重要性
varImpPlot(rf_model, main = "RandomForest特征重要性")

# ---------- 12. 模型性能对比 ----------
cat("\n========== 模型性能对比汇总 ==========\n")
comparison_df <- data.frame(
  Model = c("AdaBoost", "GBDT", "XGBoost", "RandomForest"),
  Precision = c(ada_best_metrics["Precision"], 
                gbm_best_metrics["Precision"],
                xgb_best_metrics["Precision"], 
                rf_best_metrics["Precision"]),
  Recall = c(ada_best_metrics["Recall"], 
             gbm_best_metrics["Recall"],
             xgb_best_metrics["Recall"], 
             rf_best_metrics["Recall"]),
  F1 = c(ada_best_metrics["F1"], 
         gbm_best_metrics["F1"],
         xgb_best_metrics["F1"], 
         rf_best_metrics["F1"]),
  AUC = c(ada_best_metrics["AUC"], 
          gbm_best_metrics["AUC"],
          xgb_best_metrics["AUC"], 
          rf_best_metrics["AUC"])
)

print(round(comparison_df[, -1], 4), row.names = FALSE)
print(comparison_df)

# 绘制性能对比图
comparison_long <- comparison_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "四种集成学习算法性能对比", x = "模型", y = "指标值") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ---------- 13. ROC曲线对比 ----------
roc_ada <- roc(valid_data$Survived, ada_prob)
roc_gbm <- roc(valid_data$Survived, gbm_prob)
roc_xgb <- roc(y_valid, xgb_prob)
roc_rf <- roc(valid_data$Survived, rf_prob)

plot(roc_ada, col = "red", lwd = 2, main = "ROC曲线对比")
plot(roc_gbm, col = "blue", lwd = 2, add = TRUE)
plot(roc_xgb, col = "green", lwd = 2, add = TRUE)
plot(roc_rf, col = "purple", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c("AdaBoost", "GBDT", "XGBoost", "RandomForest"),
       col = c("red", "blue", "green", "purple"), lwd = 2)

# ---------- 14. 测试集预测（使用最优XGBoost模型） ----------
x_test <- model.matrix(~ . - 1, test_processed)
final_prob <- predict(xgb_model, x_test)
final_pred <- ifelse(final_prob > 0.5, 1, 0)

# 生成提交文件
submission <- data.frame(PassengerId = test$PassengerId, Survived = final_pred)
write.csv(submission, "titanic_submission.csv", row.names = FALSE)
cat("\n预测结果已保存至 titanic_submission.csv\n")