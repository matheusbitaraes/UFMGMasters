rm(list=ls())
graphics.off()

source('perceptron.R')
source('votedPerceptron.R')
source('commiteePerceptron.R')
source('svm_model.R')
source('ggclass_model.R')
library(caret)
library(mlbench)

# MAIN
multiple_evaluations <- function(X, Y, num_eval){

  acc1 <- matrix(0, nrow=num_eval, ncol=1)
  acc2 <- matrix(0, nrow=num_eval, ncol=1)
  acc3 <- matrix(0, nrow=num_eval, ncol=1)
  acc4 <- matrix(0, nrow=num_eval, ncol=1)
  acc5 <- matrix(0, nrow=num_eval, ncol=1)
  t1 <- matrix(0, nrow=num_eval, ncol=1)
  t2 <- matrix(0, nrow=num_eval, ncol=1)
  t3 <- matrix(0, nrow=num_eval, ncol=1)
  t4 <- matrix(0, nrow=num_eval, ncol=1)
  t5 <- matrix(0, nrow=num_eval, ncol=1)
  for (i in 1:num_eval){
    print(sprintf("Iteration - %s", i))
    ts1 <- Sys.time()
    s1 <- eval_perceptron(X, Y, should_plot_matrix=FALSE)
    te1 <- Sys.time()
    
    ts2 <- Sys.time()
    s2 <- eval_svm(X, Y, should_plot_matrix=FALSE)
    te2 <- Sys.time()
    
    ts3 <- Sys.time()
    s3 <- eval_gg(X, Y, should_plot_matrix=FALSE)
    te3 <- Sys.time()
    
    ts4 <- Sys.time()
    s4 <- eval_com_perceptron(X, Y, should_plot_matrix=FALSE)
    te4 <- Sys.time()
    
    ts5 <- Sys.time()
    s5 <- eval_voted_perceptron(X, Y, should_plot_matrix=FALSE)
    te5 <- Sys.time()
    
    acc1[i] <- s1[1]
    t1[i] <- te1 - ts1
    
    acc2[i] <- s2[1]
    t2[i] <- te2 - ts2
    
    acc3[i] <- s3[1]
    t3[i] <- te3 - ts3
    
    acc4[i] <- s4[1]
    t4[i] <- te4 - ts4
    
    acc5[i] <- s5[1]
    t5[i] <- te5 - ts5
    
    # print(sprintf("perc: %s | com perc: %s | voted perc: %s \n\n", s1[1], s2[1], s3[1]))
  }
  
  return(list(data.frame("perceptron" = acc1, "svm" = acc2, "gg_classification" = acc3, "perceptron_commitee" = acc4, "voted_perceptron" = acc5),
              data.frame("perceptron" = t1, "svm" = t2, "gg_classification" = t3, "perceptron_commitee" = t4, "voted_perceptron" = t5)))
}

evaluate_dataset <- function(X, Y, name, num_exec){
  results <- multiple_evaluations(X, Y, num_exec)
  accs <- results[[1]]
  times <- results[[2]]
  
  print(summary(accs))
  print(sprintf("sd perceptron (%s)",sd(accs$perceptron)))
  print(sprintf("sd svm (%s)",sd(accs$svm)))
  print(sprintf("sd gg_classification (%s)",sd(accs$gg_classification)))
  print(sprintf("sd perceptron_commitee (%s)",sd(accs$perceptron_commitee)))
  print(sprintf("sd voted_perceptron (%s)",sd(accs$voted_perceptron)))

  par(mfrow=c(2,2))
  
  # Boxplot
  par(mar=c(4,10,1,1))
  boxplot(accs, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Acurácias (%s)",name))
  
  # Boxplot
  par(mar=c(4,10,1,1))
  boxplot(times, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Tempo (%s)",name))
  
  # Aplicação do kruskal e teste de nemenyi
  accs_matrix <- as.matrix(accs)
  tsutils::nemenyi(accs_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main="Teste de Nemenyi (Acurácias)") # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
  
  # Aplicação do kruskal e teste de nemenyi
  time_matrix <- as.matrix(times)
  tsutils::nemenyi(time_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main="Teste de Nemenyi (Tempo)") # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
}

######################################### TWO GAUSSIAN DATASET ######################################### 
nc = 500
xc1 <- matrix(0.35 * rnorm(nc) + 2, ncol = 2)
xc2 <- matrix(0.35 * rnorm(nc) + 4, ncol = 2)
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)
plot(xc1[,1], xc1[,2], col="blue", xlim=c(1, 5), ylim=c(1, 5)) # plot do dataset
points(xc2[,1], xc2[,2], col="red", pch=1) # plot do dataset
title(main="Two Gaussians Dataset")
name <- "Two Gaussian"

evaluate_dataset(X[,1:2], X[,3], name, 50)

######################################### IRIS DATASET ######################################### 
name <- "Iris Dataset"
iris.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris <- read.csv(iris.url, header=FALSE)
# pre processamento
iris$V5[iris$V5 == 'Iris-virginica'] <- 'Iris-versicolor' #juntando classe 1 com classe 2
iris$V5 <- factor(iris$V5)
iris$V5 <- as.factor(iris$V5)
names(iris)[5] <- 'y'
# plot(iris)

# converter em matrix
mat <- data.matrix(iris)
X <- scale(mat[, 1:4], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 5] - 1 # transformando em 0 e 1

evaluate_dataset(X, Y, name, 50)

######################################### WINE DATASET ######################################### 
name <- "Wine Dataset"
wine.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(wine.url, header=FALSE)
# pre-processamento
wine <- wine[1:130,]# filtra apenas dois tipos de vinho, para que fique binario
wine$V1[wine$V1 == 2] <- -1
wine$V1 <- as.factor(wine$V1)
names(wine)[1] <- 'y'

# converter em matrix
mat <- data.matrix(wine)
X <- scale(mat[, 2:14], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 1] - 1 # transformando em 0 e 1

evaluate_dataset(X, Y, name, 50)


######################################### PimaIndiansDiabetes ######################################### 
# load the dataset
name <- "Pima Indians"
pima <- data(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)[9] <- "y"

# converter em matrix
mat <- data.matrix(PimaIndiansDiabetes)
Y <- mat[, 1] - 1 # transformando entradas em 0 e 1
X <- scale(mat[, 2:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification

evaluate_dataset(X, Y, name, 50)

#########################################  CERVICAL CANCER DATASET #########################################  
name <- "Cervical Cancer Dataset"
bc.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
bc <- read.csv(bc.url, header=FALSE) 
bc <- bc[,2:11]
bc <- na.omit(bc)
bc <- droplevels(bc[!bc$V7 == '?',])
# pre processamento
bc$V11 <- as.factor(bc$V11)
names(bc)[10] <- "y"

# converter em matrix
mat <- data.matrix(bc)
Y <- mat[, 10] - 1 # subtraindo para ficar entre 0 e 1
X <- scale(mat[, 1:9], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification


evaluate_dataset(X, Y, name, 50)




#########################################  SPIRALS DATASET #########################################  
# definição do spirals
N<-1000 # numeros pares
p <- mlbench.spirals(N, 1, 0.2)
X <- p[[1]]
x1 <- X[,1]
x2 <- X[,2]
Y <- as.numeric(p[[2]]) 
name <- "Spirals Dataset"

evaluate_dataset(X, Y, name, 50)

######################################### CAESARIAN DATASET #########################################  
name <- "Caesarian Dataset"
cae <- read.csv('caesarian.csv', header=TRUE) 
# pre processamento
cae$y <- as.factor(cae$y)

# converter em matrix
mat <- data.matrix(cae)
X <- scale(mat[, 1:5], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 6]

evaluate_dataset(X, Y, name, 50)

#########################################  SONAR DATASET #########################################  
# load the dataset
name <- "Sonar Dataset"
sonar <- data(Sonar)

# converter em matrix
mat <- data.matrix(Sonar)
Y <- mat[, 61] - 1 # transformando entradas em 0 e 1
X <- scale(mat[, 0:60], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification

evaluate_dataset(X, Y, name, 50)



