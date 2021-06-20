setwd("C:/Users/SMR_l/OneDrive/Documents/INFO411/group")
##rawData <- read.table("aps_failure_training_set.csv", skip=16)
##rawData <- read.csv("aps_failure_training_set.csv")

#-----------------------------MLP----------------------------------------------#

dataTargets <- decodeClassLabels(data[,1])
dataValues <- data[,2:107]
#split dataset into traing and test set
trainset <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.4)
trainset <- normTrainingAndTestSet(trainset)

#Create mlp Model
mlpModel <- mlp(trainset$inputsTrain, trainset$targetsTrain, size=10, learnFuncParams=c(0.02), maxit= 500, inputsTest=trainset$inputsTest, targetsTest=trainset$targetsTest)
predictTestSet <- predict(mlpModel,trainset$inputsTest)

MLP_test_cm <- confusionMatrix(trainset$targetsTest,predictTestSet)

par(mfrow=c(2,2))
plotIterativeError(mlpModel)
plotRegressionError(predictTestSet[,2], trainset$targetsTest[,2])
plotROC(fitted.values(mlpModel)[,2], trainset$targetsTrain[,2])
plotROC(predictTestSet[,2], trainset$targetsTest[,2])


#check accuracy of the model predicted output against test dataset
mlpAccuracy <- sum(diag(MLP_test_cm)) / sum(MLP_test_cm)
cat("MLP accuracy : ", mlpAccuracy)

MLP_result <- confusionMatrix(predictTestSet, trainset$targetsTest)
MLP_result
MLP_tc <- total_cost(MLP_result)
cat("MLP total cost : ",MLP_tc)
