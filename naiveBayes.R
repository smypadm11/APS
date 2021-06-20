setwd("/Users/gaoanqi/Desktop/edu/911/ass")
data<-read.csv("aps.csv")
library(dplyr)
library(caret)
data$class <- as.factor(data$class)
mv_rate<-c()# to store the index where NA > 5%
n<-1
for(i in 2:171){
data[,i]<-as.numeric(as.character(data[,i]))
mv<-sum(is.na(data[,i]))
rate<-mv/60000
if(rate>0.05){ 
mv_rate[n]<- i
n<-n+1
} }
data<-data[-mv_rate] #(129)
# Change missing values to the mean of the column
for(i in 2:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}
# Split dataset 60% train and 40 % test
set.seed(3456)
trainIndex <- createDataPartition(data$class, p = .6, list = FALSE, times = 1)''
train <- data[ trainIndex,]
test <- data[-trainIndex,]

#naiveBayes
library(e1071)
library(cluster)
library(MASS)
nb<-naiveBayes(class~.,data=train)
pre<-predict(nb,test)
cm<-table(pre,test$class)
#confusion matrix
sum(diag(cm))/sum(cm)#acc:0.9659
#precision
cm[2,2]/(cm[2,1]+cm[2,2])#0.31
#recall
cm[2,2]/(cm[1,2]+cm[2,2])#0.8252