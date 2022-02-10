
library(kernlab)
eval_gg <- function(X, Y, should_plot_matrix=FALSE){
  # separação em treinamento e teste
  test_perc <- 0.3
  data_ <- as.data.frame(X)
  data_$y <- Y
  testIndexes <- createDataPartition(data_$y, p = test_perc, list = FALSE)
  x_test <- X[ testIndexes,]
  y_test <- Y[ testIndexes]
  data_test <- data_[ testIndexes,]
  
  # treinamento do perceptron
  svmKernel <- 'svmRadial'
  control <- trainControl(method="repeatedcv", number=10, repeats=4)
  trained_model <- train(y ~ .,data=data_[-testIndexes, ], method=svmKernel, trControl=control, sigma=1)
  
  model <- trained_model$finalModel
  
  y_pred_svm <- kernlab::predict(model, data_test[,names(data_test) != "y"], type='response')
  cm_svm <- confusionMatrix(table(round(y_pred_svm), as.matrix(data_test$y)))
  SMV_acc <- cm_svm$overall["Accuracy"]
  
  return (c(SMV_acc))
}


### FUNÇÕES PARA TESTAR O MÉTODO ###

# # criação de dataset com duas distribuições normais
# xc1 <- matrix(rnorm(nc * 2), ncol = 2)*s1 + t(matrix((c(2,2)),ncol=nc,nrow=2))
# xc2 <- matrix(rnorm(nc * 2), ncol = 2)*s2 + t(matrix((c(4,4)),ncol=nc,nrow=2))
# xc1 <- cbind(xc1, rep(0, times = nc/2))
# xc2 <- cbind(xc2, rep(1, times = nc/2))
# X <- rbind(xc1, xc2)
# Y <- X[,3]
# X <- X[,1:2]
# 
# 
# test_perc <- 0.3
# data_ <- as.data.frame(X)
# data_$y <- Y
# 
# testIndexes <- createDataPartition(data_$y, p = test_perc, list = FALSE)
# x_test <- X[ testIndexes,]
# y_test <- Y[ testIndexes]
# data_test <- data_[ testIndexes,]
# 
# # treinamento do perceptron
# svmKernel <- 'svmRadial'
# control <- trainControl(method="repeatedcv", number=10, repeats=4)
# trained_model <- train(y ~ .,data=data_[-testIndexes, ], method=svmKernel, trControl=control, sigma=1)
# 
# model <- trained_model$finalModel
# 
# y_pred_svm <- kernlab::predict(model, data_test[,names(data_test) != "y"], type='response')
# cm_svm <- confusionMatrix(table(round(y_pred_svm), as.matrix(data_test$y)))
# SMV_acc <- cm_svm$overall["Accuracy"]
# 
# 
# print(c("acc:", SMV_acc))
# 
# # matriz de confusão
# lvs <- c("1", "0")
# truth <-  factor(y_test, levels = rev(lvs))
# pred <- factor(ypred, levels = rev(lvs))
# 
# xtab <- table(pred, truth)
# # load Caret package for computing Confusion matrix
# library(caret)
# confusionMatrix(xtab)
# 
# seqi<-seq(0,6, 0.1)
# seqj<-seq(0,6, 0.1)
# M<-matrix(0,nrow=length(seqi), ncol=length(seqj))
# ci<-0
# for(i in seqi){
#   ci<-ci+1
#   cj<-0
#   for(j in seqj){
#     cj<-cj+1
#     x<-c(i,j)
#     M[ci,cj]<-yperceptron(x, w, -1)
#   }
# }
# 
# plot(xc1[,1], xc1[,2], col='red', xlim = c(0,6), ylim=c(0,6), xlab='x_1', ylab='x_2')
# par(new=T)
# plot(xc2[,1], xc2[,2], col='blue', xlim = c(0,6), ylim=c(0,6), xlab='', ylab='')
# par(new=T)
# contour(seqi, seqj, M, xlim=c(0,6), ylim=c(0,6))
# title("Região de Separação (2D)")
# 
# persp3D(seqi,seqj,M,counter=T, theta=55, phi=30, r=40, d=0.1, expand=0.5,
#         ltheta=90, lphi=180, shade=0.4, ticktype="detailed", nticks=5)
# title("Região de Separação (3D)")