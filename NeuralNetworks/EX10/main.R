# boston housing
rm(list=ls())

library("RSNNS")

library(mlbench)
library(MLDataR)

multiple_evaluations <- function(X, Y, num_eval){
  err1 <- matrix(0, nrow=num_eval, ncol=1)
  err2 <- matrix(0, nrow=num_eval, ncol=1)
  err3 <- matrix(0, nrow=num_eval, ncol=1)
  for (i in 1:num_eval){
    m1 <- mlp(X, Y, size=20, maxit=200, initFunc = "Randomize_Weigths",
                     initFuncParams = c(-0.3,0.3), learnFunc = "Rprop",
                     learnFuncParams = c(0.1, 0.1), updateFunc = "Topological_Order",
                     updateFuncParams = c(0), hiddenActFunc = "Act_Logistic",
                     shufflePatterns = TRUE, linOut = TRUE)
    m2 <- mlp(X, Y, size=50, maxit=200, initFunc = "Randomize_Weigths",
              initFuncParams = c(-0.3,0.3), learnFunc = "Rprop",
              learnFuncParams = c(0.1, 0.1), updateFunc = "Topological_Order",
              updateFuncParams = c(0), hiddenActFunc = "Act_Identity",
              shufflePatterns = TRUE, linOut = TRUE)
    m3 <- mlp(X, Y, size=80, maxit=200, initFunc = "Randomize_Weigths",
              initFuncParams = c(-0.3,0.3), learnFunc = "Rprop",
              learnFuncParams = c(0.1, 0.1), updateFunc = "Topological_Order",
              updateFuncParams = c(0), hiddenActFunc = "Act_Logistic",
              shufflePatterns = TRUE, linOut = TRUE)
    
    err1[i] <- m1$IterativeFitError[m1$maxit]
    err2[i] <- m2$IterativeFitError[m2$maxit]
    err3[i] <- m3$IterativeFitError[m3$maxit]
  }
  errs <- data.frame("model1" = err1,
                     "model2" = err2,
                     "model3" = err3)

  return(list(errs))
}

evaluate_dataset <- function(X, Y, name, num_exec){
  results <- multiple_evaluations(X, Y, num_exec)
  accs <- results[[1]]
  
  print(name)
  n <- names(accs)
  accs_matrix <- as.matrix(accs)
  for (i in 1:ncol(accs_matrix)){
    avg <- round(mean(accs_matrix[,i]), digits=4)
    std <- round(sd(accs_matrix[,i]), digits=4)
    print(sprintf("%s: %s +- %s", n[i], avg, std))
  }
  
  # par(mfrow=c(1,2))
  
  # Boxplot
  par(mar=c(2,16,1,1))
  boxplot(accs, data=data,
          las = 1,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Erros (%s)",name))
  
  # Aplicação do kruskal e teste de nemenyi
  accs_matrix <- as.matrix(accs)
  tsutils::nemenyi(accs_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main=sprintf("Teste de Nemenyi Erros (%s)", name)) # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
}



name <- "Boston Housing"
data(BostonHousing)
x_class = BostonHousing[, -c(14)]
X = data.matrix(x_class)
dimnames(X) = NULL
Y = as.numeric(BostonHousing$medv)
evaluate_dataset(X, Y, name, 10)


#

# Statlog heart
heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
name <- "Statlog (heart)"
hd <- na.omit(heart.data)
x_class = hd[, -c(14)]
X = data.matrix(na.omit(x_class) )
X = scale(X, center = TRUE, scale = TRUE)
dimnames(X) = NULL
Y = as.numeric(hd$num)
evaluate_dataset(X, Y, name, 10)



# escalonar os resultados
