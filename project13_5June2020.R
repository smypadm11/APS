setwd("D:/IVAN/6th session/INFO411 - Data Mining and Knowledge Discovery/Project")
#setwd("/Users/gaoanqi/Desktop/edu/911/ass")
#data<-read.csv("aps.csv")

# install library
# install.packages("ranger")
# install.packages("unbalanced")
# install.packages("naivebayes")
# install.packages("VIM")

# import library
library(randomForest)
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
library(RSNNS)

# load training data
data <- read.csv("aps_failure_training_set.csv", stringsAsFactors = FALSE, header = TRUE)

# Change class column into factor
data$class <- as.factor(data$class)

# Change column 2-171 into numeric and calculate % of missing value for each column 
mv_rate<-c()# to store the index where NA > 5%
n<-1
for(i in 2:170){
  data[,i]<-as.numeric(as.character(data[,i]))
  mv<-sum(is.na(data[,i]))
  rate<-mv/60000
  if(rate>0.05){ 
    mv_rate[n]<- i
    n<-n+1
  } 
}

# Detect missing values
str(data)
print(sum(is.na(data)))
head(data, n=10)

# if NA > 5% delete the entire column
data <- data[-mv_rate] #(129)

# remove columns where over 90% of instance values are zero
col_zero <- lapply(data, function(col){length(which(col==0))/length(col)})
data <- data[, !(names(data) %in% names(col_zero[lapply(col_zero, function(x) x) > 0.9]))]

# visualize the missing value patterns
miss <- aggr(data)

# Change missing values to the mean of the column
for(i in 2:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}

# Checking missing values again
print(sum(is.na(data)))

# Visualize the ditribution of the dataset
qplot(as.factor(data$class), xlab = "class")

# Split dataset 60% train and 40 % test
set.seed(3456)
trainIndex <- createDataPartition(data$class, p = .6, list = FALSE, times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]

# Calculate total cost
total_cost <- function(result){
  costs <- matrix(c(0, 10, 500, 0), 2)
  return(sum(result * costs))
}

#-----------------------------Random Forest----------------------------------------------#
# Random forest model fitting
rf_classifier <- randomForest(class ~ ., data=train, ntree=100, mtry=2, importance=TRUE)
varImpPlot(rf_classifier)
plot(rf_classifier)

# Predict the class with Random Forest
rf_predict <- predict(rf_classifier, test)
test_cm <- table(rf_predict, test$class)
rf_result <- confusionMatrix(rf_predict, test$class)
rf_result

as.numeric(unlist(rf_classifier))
plot(margin(rf_classifier))
rf_classifier$importance

# Accuracy for Random Forest
rf_accuracy <- sum(diag(rf_result)/nrow(test))
cat("Random Forest accuracy : ", rf_accuracy)

rf_tc <- total_cost(rf_result)
cat("Random Forest total cost : ",rf_tc)

print(rf_classifier)
summary(rf_classifier)

#precision
rf_precision <- rf_result[2,2]/(rf_result[2,1]+rf_result[2,2])
cat("Random Forest precision : ",rf_precision)

#recall
rf_recall <- rf_result[2,2]/(rf_result[1,2]+rf_result[2,2])
cat("Random Forest recall : ",rf_recall)

#f1-score
rf_f1 <- (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)
cat("Random Forest f1-score : ",rf_f1)
#-----------------------------MLP----------------------------------------------#

dataTargets <- decodeClassLabels(data[,1])
dataValues <- data[,2:107]
#split dataset into traing and test set
trainset <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.4)
trainset <- normTrainingAndTestSet(trainset)

#Create mlp Model
mlpModel <- mlp(trainset$inputsTrain, trainset$targetsTrain, size=10, learnFuncParams=c(0.02), maxit= 500, inputsTest=trainset$inputsTest, targetsTest=trainset$targetsTest)

#predict with MLP model
predictTestSet <- predict(mlpModel,trainset$inputsTest)
MLP_test_cm <- confusionMatrix(trainset$targetsTest,predictTestSet)
MLP_test_cm

#visualization
par(mfrow=c(2,2))
plotIterativeError(mlpModel)
plotRegressionError(predictTestSet[,2], trainset$targetsTest[,2])
plotROC(fitted.values(mlpModel)[,2], trainset$targetsTrain[,2])
plotROC(predictTestSet[,2], trainset$targetsTest[,2])


#check accuracy of the model predicted output against test dataset
mlpAccuracy <- sum(diag(MLP_test_cm)) / sum(MLP_test_cm)
cat("MLP accuracy : ", mlpAccuracy)

MLP_tc <- total_cost(MLP_test_cm)
cat("MLP total cost : ",MLP_tc)

#-----------------------------Resultt Summarize----------------------------------------------#
tc <- as.data.frame(cbind(c("Random Forest", "MLP"), 
                          c(rf_tc, MLP_tc),
                          c(rf_accuracy,mlpAccuracy)))

colnames(tc) <- c("Model","Total_Cost","Accuracy")

arrange(tc, Total_Cost)
