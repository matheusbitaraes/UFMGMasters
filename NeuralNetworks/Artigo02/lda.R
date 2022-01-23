library(MASS)
library(caret)

eval_lda <- function(X, Y, should_plot_matrix=FALSE, i = 1){
  set.seed(i)
  nc <- nrow(X)
  suffled_indexes <- sample(nc)
  train_size <- floor(nc * 0.70)
  x_train <- X[suffled_indexes[1:train_size],]
  y_train <- Y[suffled_indexes[1:train_size]]
  x_test <- X[suffled_indexes[(train_size+1):nc],]
  y_test <- Y[suffled_indexes[(train_size+1):nc]]
  
  # treinamento do perceptron
  model <- lda(x_train, y_train)
  
  # y_class_onehot = onehot_encode(y_train) # class labels should begin from 0
  # sol <- elm_train(x_train, y_class_onehot, nhid = 20, actfun = 'relu')
  
  # acurácia 
  y_pred <- predict(model, x_test)$class
  # y_pred = elm_predict(sol, x_test, normalize = TRUE)
  # ypred <- y_pred 
  # ypred[ypred>=0] <- 0
  # ypred[ypred<0] <- 1
  
  # matriz de confusão
  lvs <- c("0", "1")
  truth <-  factor(y_test, levels = rev(lvs))
  pred <- factor(y_pred, levels = rev(lvs))
  
  xtab <- table(pred, truth)
  cm <- confusionMatrix(xtab)
  if (should_plot_matrix){
    print(cm)
  }
  return (c(cm$overall[1]))
}


# data(ionosphere, package = 'KernelKnn')
# x_class = ionosphere[, -c(2, ncol(ionosphere))]
# x_class = as.matrix(x_class)
# dimnames(x_class) = NULL
# y_class = as.numeric(ionosphere[, ncol(ionosphere)]) - 1
# 
# X = x_class
# Y = y_class
# should_plot_matrix=FALSE
# i = 1


