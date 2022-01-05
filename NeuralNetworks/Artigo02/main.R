rm(list=ls())
graphics.off()

source('rbf.R')
source('elm.R')
library(caret) 

# escolher os metodos de avaliação de partição do ClusterCrit do R
# estudar o conjunto de dados e aplicar os metodos de avaliação de qualidade de cluster
# selecionar variaveis ue vai melhorar o indice de qualidade do cluster

# clusterizar as features de acordo com regras x, y e z e depois avalia qual metrica ficou melhor, pelo metodo de avaliação de clusteres.
# x = kmeans
# y = ?
# z = ?

# aí joga no ELM e RBF para ver a qualidade. E compara com o que se acha na literatura


# MAIN
multiple_evaluations <- function(X, Y, num_eval){
  
  m1_acc <- matrix(0, nrow=num_eval, ncol=1)
  m1_time <- matrix(0, nrow=num_eval, ncol=1)
  m2_acc <- matrix(0, nrow=num_eval, ncol=1)
  m2_time <- matrix(0, nrow=num_eval, ncol=1)
  for (i in 1:num_eval){
    print(sprintf("Iteration - %s", i))
    m1_start_time <- Sys.time()
    s1 <- eval_elm(X, Y, should_plot_matrix=FALSE)
    m1_end_time <- Sys.time()
    
    
    m2_start_time <- Sys.time()
    s2 <- eval_rbf(X, Y, should_plot_matrix=FALSE)
    m2_end_time <- Sys.time()
    
    m1_acc[i] <- s1[1]
    m1_time[i] <- m1_end_time - m1_start_time
    
    m2_acc[i] <- s2[1]
    m2_time[i] <- m2_end_time - m2_start_time
    
    print(sprintf("RBF: %s | ELM: %s \n\n", s1[1], s2[1]))
  }
  
  return(list(data.frame("RBF acc" = m1_acc, "ELM acc" = m2_acc),
              data.frame("RBF time" = m1_time, "ELM time" = m2_time)))
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


######################################### IONOSPHERE DATASET ######################################### 
name <- "Ionosphere Dataset"
data(ionosphere, package = 'KernelKnn')
x_class = ionosphere[, -c(2, ncol(ionosphere))]
X = as.matrix(x_class)
dimnames(X) = NULL
Y = as.numeric(ionosphere[, ncol(ionosphere)]) - 1
evaluate_dataset(X, Y, name, 50)


# ######################################### WINE DATASET ######################################### 
# name <- "Wine Dataset"
# wine.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# wine <- read.csv(wine.url, header=FALSE)
# # pre-processamento
# wine <- wine[1:130,]# filtra apenas dois tipos de vinho, para que fique binario
# wine$V1[wine$V1 == 2] <- -1
# wine$V1 <- as.factor(wine$V1)
# names(wine)[1] <- 'y'
# 
# # converter em matrix
# mat <- data.matrix(wine)
# X <- scale(mat[, 2:14], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
# Y <- mat[, 1] - 1 # transformando em 0 e 1
# 
# evaluate_dataset(X, Y, name, 50)
# 
# 
# 
# ######################################### PimaIndiansDiabetes ######################################### 
# # load the dataset
# name <- "Pima Indians"
# pima <- data(PimaIndiansDiabetes)
# names(PimaIndiansDiabetes)[9] <- "y"
# 
# # converter em matrix
# mat <- data.matrix(PimaIndiansDiabetes)
# Y <- mat[, 1] - 1 # transformando entradas em 0 e 1
# X <- scale(mat[, 2:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
# 
# evaluate_dataset(X, Y, name, 50)
# 
# #########################################  CERVICAL CANCER DATASET #########################################  
# name <- "Cervical Cancer Dataset"
# bc.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
# bc <- read.csv(bc.url, header=FALSE) 
# bc <- bc[,2:11]
# bc <- na.omit(bc)
# bc <- droplevels(bc[!bc$V7 == '?',])
# # pre processamento
# bc$V11 <- as.factor(bc$V11)
# names(bc)[10] <- "y"
# 
# # converter em matrix
# mat <- data.matrix(bc)
# Y <- mat[, 10] - 1 # subtraindo para ficar entre 0 e 1
# X <- scale(mat[, 1:9], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
# 
# 
# evaluate_dataset(X, Y, name, 50)
# 
# 
# 
