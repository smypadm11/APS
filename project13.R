setwd("D:/IVAN/6th session/INFO411 - Data Mining and Knowledge Discovery/Project")

# install library
# install.packages("ranger")
# install.packages("unbalanced")
# install.packages("naivebayes")
# install.packages("VIM")

# import library
library(data.table)
library(dplyr)
library(ranger)
library(caret)
library(unbalanced)
library(naivebayes)
library(rpart)
library(mice)
library(ggplot2)
library(VIM)
library(pROC)

# load training data
data <- read.csv("aps_failure_training_set.csv", stringsAsFactors = FALSE, header = TRUE)

# Change class column into factor
data$class <- as.factor(data$class)

# Change column 2-171 into numeric
sapply(2:ncol(data), function(i) {
  data[, i] <<- as.numeric(data[, i])
})

# Detect missing values
str(data)
print(sum(is.na(data)))
head(data, n=10)

# Change missing values to the mean of the column
for(i in 2:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}

# Checking missing values again
print(sum(is.na(data)))
head(data, n=10)

# Visualize the ditribution of the dataset
qplot(as.factor(data$class), xlab = "class")

# Split dataset 60% train and 40 % test
set.seed(3456)
trainIndex <- createDataPartition(data$class, p = .6, list = FALSE, times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]

# Perform training:
rf_classifier = randomForest(Species ~ ., data=training, ntree=100, mtry=2, importance=TRUE)
