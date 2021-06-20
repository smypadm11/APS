setwd("D:/UOW session 3/Data Mining/Assignment/Project")
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
library("partykit")

# load training data
data <- read.csv("aps_failure_training_set.csv", stringsAsFactors = FALSE, header = TRUE)
head(data)
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
attach(data)

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

# Fit decision tree to training dataset
dtree_classifier <- rpart(class ~., data = train, method = "class")
jpeg("dtree.jpeg", width = 800)
print(dtree_classifier)
plot(dtree_classifier, uniform=TRUE, margin=0.2)
text(dtree_classifier, use.n=TRUE, all=TRUE, cex=.9)
dev.off()

dt_predict <- predict(dtree_classifier, test, type = "class")

dt_result <- confusionMatrix(dt_predict, test$class)
dt_result
res <- table(predict(dtree_classifier, test, type = "class"), test$class)
res
res.accuracy <- sum(diag(dt_result))/nrow(test)
res.accuracy

dt_tc <- total_cost(dt_result)
dt_tc
