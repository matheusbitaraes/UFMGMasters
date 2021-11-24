rm(list=ls())
graphics.off()

source('perceptron.R')
source('votedPerceptron.R')
source('fmpImaPerceptron.R')


# MAIN
multiple_evaluations <- function(X, Y, num_eval){

  perc_acc <- matrix(0, nrow=num_eval, ncol=1)
  perc_time <- matrix(0, nrow=num_eval, ncol=1)
  fmp_acc <- matrix(0, nrow=num_eval, ncol=1)
  fmp_time <- matrix(0, nrow=num_eval, ncol=1)
  vp_acc <- matrix(0, nrow=num_eval, ncol=1)
  vp_time <- matrix(0, nrow=num_eval, ncol=1)
  for (i in 1:num_eval){
    perc_start_time <- Sys.time()
    s1 <- eval_perceptron(X, Y, should_plot_matrix=FALSE)
    perc_end_time <- Sys.time()
    
    fmp_start_time <- Sys.time()
    s2 <- eval_fmp_ima_perceptron(X, Y, should_plot_matrix=FALSE)
    fmp_end_time <- Sys.time()
    
    vp_start_time <- Sys.time()
    s3 <- eval_voted_perceptron(X, Y, should_plot_matrix=FALSE)
    vp_end_time <- Sys.time()
    
    perc_acc[i] <- s1[1]
    perc_time[i] <- perc_end_time - perc_start_time
    
    fmp_acc[i] <- s2[1]
    fmp_time[i] <- fmp_end_time - fmp_start_time
    
    vp_acc[i] <- s3[1]
    vp_time[i] <- vp_end_time - vp_start_time
  }
  
  return(list(data.frame("perceptron" = perc_acc, "fmp_ima" = fmp_acc, "voted_perceptron" = vp_acc),
              data.frame("perceptron" = perc_time, "fmp_time" = fmp_acc, "voted_perceptron" = vp_time)))
}

evaluate_dataset <- function(X, Y, name, num_exec){
  results <- multiple_evaluations(X, Y, num_exec)
  accs <- results[[1]]
  times <- results[[2]]
  
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
Y <- mat[, 5]

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
Y <- mat[, 1]

evaluate_dataset(X, Y, name, 5, 100)

######################################### PimaIndiansDiabetes ######################################### 
# load the dataset
name <- "Pima Indians"
pima <- data(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)[9] <- "y"

# converter em matrix
mat <- data.matrix(PimaIndiansDiabetes)
X <- scale(mat[, 1:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 9]

evaluate_dataset(X, Y, name, 5, 100)

#########################################  CERVICAL CANCER DATASET #########################################  
# definição do spirals
N<-1000 # numeros pares
p <- mlbench.spirals(N, 1, 0.15)
X <- p[[1]]
x1 <- X[,1]
x2 <- X[,2]
Y <- p[[2]]
name <- "Spirals Dataset"

evaluate_dataset(X, Y, name, 5, 100)

######################################### CAESARIAN DATASET #########################################  
name <- "Caesarian Dataset"
cae <- read.csv('caesarian.csv', header=TRUE) 
# pre processamento
cae$y <- as.factor(cae$y)

# converter em matrix
mat <- data.matrix(cae)
X <- scale(mat[, 1:5], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 6]

evaluate_dataset(X, Y, name, 4, 100)

#########################################  CERVICAL CANCER DATASET #########################################  
name <- "Breast Cancer Dataset"
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
X <- scale(mat[, 1:9], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 10]

evaluate_dataset(X, Y, name, 5, 100)

